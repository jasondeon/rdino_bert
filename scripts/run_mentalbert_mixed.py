"""Matched MentalBERT experiment using existing mixed-study subject folds.

The completed LOSO runner remains unchanged for reproducibility. This runner
reuses its model/data/inference helpers and preserves its optimization loop,
adding per-study inner validation metrics for equal-study epoch selection.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from clinical_evaluation import regression_metrics, stable_id
from mentalbert_evaluation import (
    adaptation_state, aggregate_chunks, collate_recordings, restore_adaptation,
    sample_recording_windows, target_standardization,
)
from mixed_evaluation import evaluate_mixed, mixed_partitions, select_mixed_epoch, study_metrics
from scripts import run_mentalbert_loso as loso
from scripts.run_mentalbert_loso import (
    atomic_json, atomic_checkpoint, fingerprint, guard_configuration,
    make_model, autocast_context, predict_windows, recording_predictions,
)
from scripts.build_evaluation_bundle import digest, write_csv
from scripts.score_evaluation_bundle import read_csv


def fit_inner_stage(directory, train, validation, grouped, tokenizer, config, device, epochs, seed):
    directory.mkdir(parents=True, exist_ok=True)
    specification = {
        "experiment": fingerprint(config), "epochs": epochs, "seed": seed,
        "train_ids": [r["recording_id"] for r in train],
        "validation_ids": [r["recording_id"] for r in validation],
    }
    guard_configuration(directory / "stage_config.json", specification)
    signature = fingerprint(specification)
    checkpoint_path = directory / "last_checkpoint.pt"
    complete_path = directory / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete["signature"] != signature or digest(checkpoint_path) != complete["checkpoint_sha256"]:
            raise ValueError(f"Completed stage integrity mismatch: {directory}")
        if digest(directory / "history.json") != complete["history_sha256"]:
            raise ValueError("Completed stage history changed")
        print(f"Skipping completed stage: {directory}", flush=True)
        return json.loads((directory / "history.json").read_text())
    standardization = target_standardization(train)
    model = make_model(config, device, seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, start_epoch = [], 1
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["standardization"] != standardization:
            raise ValueError("Resume checkpoint does not match this stage")
        restore_adaptation(model, checkpoint["adaptation"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        del checkpoint
        print(f"Resuming {directory} at epoch {start_epoch}", flush=True)
    else:
        print(f"Starting {directory}: {len(train)} training recordings, {sum(p.numel() for p in parameters):,} trainable parameters", flush=True)
    validation_windows = [w for r in validation for w in grouped[r["recording_id"]]]
    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        model.train()
        sampled = sample_recording_windows(train, grouped, config["windows_per_recording"], seed, epoch)
        batch_size = config["recordings_per_batch"]
        total_loss, total_recordings = 0.0, 0
        for offset in range(0, len(sampled), batch_size):
            examples = sampled[offset:offset + batch_size]
            tokens, owners, weights, targets = collate_recordings(examples, tokenizer, standardization)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            owners, weights, targets = owners.to(device), weights.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config["precision"]):
                chunk_predictions = model(tokens)
                prediction = aggregate_chunks(chunk_predictions.float(), owners, weights, len(examples))
                loss = torch.nn.functional.mse_loss(prediction, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss: {directory}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(examples)
            total_recordings += len(examples)
        training_mse = total_loss / total_recordings
        if validation and (epoch % config["validation_interval"] == 0 or epoch == epochs):
            window_prediction = predict_windows(
                model, validation_windows, tokenizer, standardization, device,
                config["inference_batch_size"], config["precision"],
            )
            predictions = recording_predictions(validation, grouped, window_prediction)
            metrics = regression_metrics([float(r["regression_label"]) for r in validation], predictions)
            metrics["by_study"] = study_metrics(validation, predictions)
            metrics["macro_study_rmse"] = sum(m["rmse"] for m in metrics["by_study"].values()) / len(metrics["by_study"])
            history.append({"epoch": epoch, "training_mse_standardized": training_mse, **metrics})
            print(f"{directory.name} epoch {epoch}/{epochs}: train MSE={training_mse:.4f}, validation RMSE={metrics['rmse']:.3f}, R2={metrics['r2']}", flush=True)
        else:
            print(f"{directory.name} epoch {epoch}/{epochs}: train MSE={training_mse:.4f}, seconds={time.monotonic()-started:.1f}", flush=True)
        atomic_checkpoint(checkpoint_path, {
            "signature": signature, "epoch": epoch,
            "adaptation": adaptation_state(model), "optimizer": optimizer.state_dict(),
            "standardization": standardization, "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "model_config": config,
        })
        atomic_json(directory / "history.json", history)
    # Also restore history.json if a process was interrupted between checkpoint
    # replacement and its human-readable history export.
    atomic_json(directory / "history.json", history)
    atomic_json(complete_path, {
        "signature": signature, "epochs": epochs,
        "checkpoint_sha256": digest(checkpoint_path),
        "history_sha256": digest(directory / "history.json"),
    })
    del model, optimizer, parameters
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history



def preflight(args):
    reference_path = args.reference_run / 'experiment_config.json'
    reference = json.loads(reference_path.read_text())
    fields = ('lora_rank', 'lora_alpha', 'normalization', 'learning_rate', 'weight_decay',
              'max_epochs', 'validation_interval', 'recordings_per_batch', 'windows_per_recording',
              'inference_batch_size', 'max_tokens', 'precision', 'seed')
    inherited = SimpleNamespace(
        **{key: reference[key] for key in fields}, bundle=args.bundle,
        model=reference['model_name'], weights_revision=reference['weights_revision'],
        allow_download=not reference['local_files_only'], study=None,
    )
    config, report, tokenizer, grouped, _ = loso.preflight(inherited)
    differences = [key for key in set(config) | set(reference) if config.get(key) != reference.get(key)]
    if differences:
        raise ValueError(f'Runtime no longer matches the reference LOSO configuration: {differences}')
    records, _, folds = mixed_partitions(args.bundle)
    if args.fold and not set(args.fold) <= set(folds):
        raise ValueError(f'Unknown fold; expected {sorted(folds)}')
    config = dict(config)
    config.update(protocol='mixed_subject', reference_config_sha256=digest(reference_path),
                  selection='pool inner OOF squared errors within each study, then minimize equal-study mean RMSE')
    root = Path(__file__).resolve().parents[1]
    config['code_sha256'] = dict(config['code_sha256'])
    for path in (Path(__file__).resolve(), root / 'mixed_evaluation.py', root / 'scripts/build_evaluation_bundle.py'):
        config['code_sha256'][str(path.relative_to(root))] = digest(path)
    report.update(protocol='mixed_subject', settings_match_reference=True,
                  outer_fits=len(folds), inner_fits=sum(len(f['inner']) for f in folds.values()),
                  folds=[{'fold': name, 'train_subjects': len({r['subject_id'] for r in fold['train']}),
                          'test_subjects': len({r['subject_id'] for r in fold['test']}),
                          'train_recordings': len(fold['train']), 'test_recordings': len(fold['test']),
                          'test_recordings_by_study': {study: sum(r['study'] == study for r in fold['test'])
                                                       for study in report['studies']}}
                         for name, fold in folds.items()])
    return config, report, tokenizer, grouped, folds


def check_complete(directory, config):
    completion_path = directory / 'complete.json'
    if not completion_path.exists():
        return False
    completion = json.loads(completion_path.read_text())
    if completion['experiment'] != fingerprint(config):
        raise ValueError(f'Completed fold configuration mismatch: {directory}')
    if completion['predictions_sha256'] != digest(directory / 'window_predictions.csv'):
        raise ValueError(f'Completed fold prediction checksum mismatch: {directory}')
    return True


def run_outer(name, fold, output, grouped, tokenizer, config, device):
    directory = output / name
    directory.mkdir(exist_ok=True)
    if check_complete(directory, config):
        print(f'Skipping completed outer fold: {name}', flush=True)
        return
    histories = []
    for inner_name, inner in fold['inner'].items():
        seed = (config['seed'] + int(stable_id(f'mixed/{name}/inner/{inner_name}')[:8], 16)) % (2**31)
        histories.append(fit_inner_stage(directory / 'inner' / inner_name,
                                         inner['train'], inner['validation'], grouped, tokenizer,
                                         config, device, config['max_epochs'], seed))
    epoch, curve = select_mixed_epoch(histories)
    atomic_json(directory / 'epoch_selection.json', {
        'selected_epoch': epoch, 'selected_at_max_epochs': epoch == config['max_epochs'],
        'selection': config['selection'], 'curve': curve,
    })
    print(f'{name}: selected {epoch} epochs using inner validation only', flush=True)
    seed = (config['seed'] + int(stable_id(f'mixed/{name}/final')[:8], 16)) % (2**31)
    loso.fit_stage(directory / 'final', fold['train'], [], grouped, tokenizer, config, device, epoch, seed)
    checkpoint = torch.load(directory / 'final' / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
    model = make_model(config, device, seed)
    restore_adaptation(model, checkpoint['adaptation'])
    windows = [w for row in fold['test'] for w in grouped[row['recording_id']]]
    prediction = predict_windows(model, windows, tokenizer, checkpoint['standardization'], device,
                                 config['inference_batch_size'], config['precision'])
    studies = {r['recording_id']: r['study'] for r in fold['test']}
    rows = [{'window_id': w.window_id, 'test_fold': name, 'study': studies[w.recording_id],
             'prediction': prediction[w.window_id]} for w in windows]
    write_csv(directory / 'window_predictions.csv', rows)
    values = recording_predictions(fold['test'], grouped, prediction)
    write_csv(directory / 'recording_predictions.csv', [
        {'recording_id': r['recording_id'], 'test_fold': name, 'study': r['study'], 'prediction': p}
        for r, p in zip(fold['test'], values, strict=True)
    ])
    atomic_json(directory / 'test_metrics.json', {
        'pooled': regression_metrics([float(r['regression_label']) for r in fold['test']], values),
        'by_study': study_metrics(fold['test'], values),
    })
    atomic_json(directory / 'complete.json', {'experiment': fingerprint(config),
                'selected_epoch': epoch, 'predictions_sha256': digest(directory / 'window_predictions.csv')})
    del checkpoint, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def write_comparison(reference_run, output, mixed_metrics, mixed_rows):
    """Compare identical recordings; do not compare mismatched historical cohorts."""
    reference_path = reference_run / 'loso_recording_predictions.csv'
    if not reference_path.exists():
        print('Reference predictions unavailable; mixed-study metrics are still complete.', flush=True)
        return
    reference = read_csv(reference_path)
    old = {r['recording_id']: r for r in reference}
    if len(old) != len(reference) or set(old) != {r['recording_id'] for r in mixed_rows}:
        raise ValueError('Reference and mixed predictions do not cover identical recordings')
    comparison, paired = [], []
    for row in mixed_rows:
        ref = old[row['recording_id']]
        if (ref['study'], ref['subject_id'], float(ref['truth'])) != (row['study'], row['subject_id'], row['truth']):
            raise ValueError('Reference metadata/targets differ from mixed predictions')
        paired.append({'recording_id': row['recording_id'], 'subject_id': row['subject_id'],
                       'study': row['study'], 'truth': row['truth'],
                       'loso_prediction': float(ref['prediction']), 'mixed_prediction': row['prediction']})
    for label in ['ALL'] + sorted({r['study'] for r in paired}):
        subset = paired if label == 'ALL' else [r for r in paired if r['study'] == label]
        y = [r['truth'] for r in subset]
        a = regression_metrics(y, [r['loso_prediction'] for r in subset])
        b = regression_metrics(y, [r['mixed_prediction'] for r in subset])
        comparison.append({'study': label, 'recordings': len(subset),
                           'loso_mae': a['mae'], 'mixed_mae': b['mae'], 'mae_gain_mixed': a['mae']-b['mae'],
                           'loso_rmse': a['rmse'], 'mixed_rmse': b['rmse'],
                           'loso_r2': a['r2'], 'mixed_r2': b['r2'],
                           'loso_mean_error': a['mean_prediction_error'], 'mixed_mean_error': b['mean_prediction_error']})
    write_csv(output / 'comparison_with_loso.csv', comparison)
    write_csv(output / 'paired_predictions.csv', paired)
    atomic_json(output / 'comparison_provenance.json', {
        'reference_predictions_sha256': digest(reference_path),
        'matched_recordings': len(paired),
        'limitations': ['Training-set size and composition differ between designs.',
                       'The designs use independent stage seeds and independently selected epochs.',
                       'A single-seed difference is not a causal estimate of the effect of study exposure.'],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('outputs/evaluation/20260913_v3'))
    parser.add_argument('--reference-run', type=Path, default=Path('outputs/mentalbert-loso-20260913'))
    parser.add_argument('--output', type=Path, default=Path('outputs/mentalbert-mixed-20260913'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fold', action='append', help='Select an outer fold, e.g. fold_0; repeat for several')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    for name in ('bundle', 'reference_run', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    for protected in (args.bundle, args.reference_run):
        if args.output == protected or protected in args.output.parents:
            parser.error('Outputs must be separate from the bundle and completed reference run')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another process is using this experiment output')
        config, report, tokenizer, grouped, folds = preflight(args)
        guard_configuration(args.output / 'experiment_config.json', config)
        atomic_json(args.output / 'preflight.json', report)
        print(json.dumps(report, indent=2), flush=True)
        if args.dry_run:
            print('CPU preflight complete; no model weights loaded or training started.')
            return
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        if config['precision'] == 'bfloat16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA device does not support bfloat16')
        torch.backends.cudnn.benchmark = False
        tokenizer.save_pretrained(args.output / 'tokenizer')
        for name in args.fold or sorted(folds):
            run_outer(name, folds[name], args.output, grouped, tokenizer, config, device)
        if all(check_complete(args.output / name, config) for name in folds):
            rows = [row for name in sorted(folds) for row in read_csv(args.output / name / 'window_predictions.csv')]
            metrics, recordings = evaluate_mixed(args.bundle, rows)
            metrics['experiment_config_sha256'] = digest(args.output / 'experiment_config.json')
            write_csv(args.output / 'mixed_window_predictions.csv', rows)
            write_csv(args.output / 'mixed_recording_predictions.csv', recordings)
            atomic_json(args.output / 'metrics.json', metrics)
            write_comparison(args.reference_run, args.output, metrics, recordings)
            print(json.dumps({'macro_study': metrics['macro_study'], 'pooled': metrics['pooled']}, indent=2))
        else:
            print('Selected folds complete. Rerun without --fold to finish the combined evaluation.')


if __name__ == '__main__':
    main()
