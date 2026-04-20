#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Experiment Runner Script for NTK-SURGERY

This script runs all experiments from the manuscript:
- Main experiments (all datasets)
- Ablation study
- Domain generalization
- Hyperparameter search

All experiments are deterministic (no random operations).

Usage:
    ./run_experiments.sh [OPTIONS]

Options:
    --experiment    Experiment type (main|ablation|domain|hyperparam|all)
    --datasets      Comma-separated list of datasets
    --device        Computing device (cpu|cuda)
    --output_dir    Output directory for results
    --resume        Resume from checkpoint if available
    --dry_run       Print commands without executing
    --help          Show this help message

Example:
    ./run_experiments.sh --experiment main --datasets CIFAR-10,CIFAR-100
    ./run_experiments.sh --experiment all --device cuda
"""

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
readonly SCRIPT_NAME="run_experiments.sh"
readonly SCRIPT_VERSION="1.0.0"
readonly DEFAULT_OUTPUT_DIR="results"
readonly DEFAULT_DEVICE="cpu"
readonly LOG_FILE="logs/experiments.log"

# Experiment types
readonly EXPERIMENT_TYPES="main,ablation,domain,hyperparam,all"

# ==============================================================================
# LOGGING FUNCTIONS
# ==============================================================================
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "INFO" "$@"
}

log_warning() {
    log "WARNING" "$@"
}

log_error() {
    log "ERROR" "$@"
}

log_success() {
    log "SUCCESS" "$@"
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
show_help() {
    cat << EOF
${SCRIPT_NAME} v${SCRIPT_VERSION}

Run NTK-SURGERY experiments from the manuscript.

Usage:
    ${SCRIPT_NAME} [OPTIONS]

Options:
    --experiment    Experiment type: ${EXPERIMENT_TYPES} (default: all)
    --datasets      Comma-separated list of datasets (default: all)
    --device        Computing device: cpu|cuda (default: ${DEFAULT_DEVICE})
    --output_dir    Output directory for results (default: ${DEFAULT_OUTPUT_DIR})
    --resume        Resume from checkpoint if available
    --dry_run       Print commands without executing
    --help          Show this help message

Examples:
    ${SCRIPT_NAME} --experiment main
    ${SCRIPT_NAME} --experiment ablation --datasets CIFAR-10
    ${SCRIPT_NAME} --experiment all --device cuda --output_dir /results

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check for required Python packages
    if ! python3 -c "import torch" &> /dev/null; then
        missing_deps+=("torch (pip install torch)")
    fi
    
    if ! python3 -c "import numpy" &> /dev/null; then
        missing_deps+=("numpy (pip install numpy)")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All dependencies available"
}

create_directories() {
    local output_dir="$1"
    
    log_info "Creating directories..."
    
    mkdir -p "${output_dir}"
    mkdir -p "${output_dir}/main_experiments"
    mkdir -p "${output_dir}/ablation"
    mkdir -p "${output_dir}/domain_generalization"
    mkdir -p "${output_dir}/hyperparameter_search"
    mkdir -p "${output_dir}/visualizations"
    mkdir -p "$(dirname "${LOG_FILE}")"
    mkdir -p "logs"
    mkdir -p "checkpoints"
    
    log_success "Directories created"
}

# ==============================================================================
# EXPERIMENT RUNNER FUNCTIONS
# ==============================================================================
run_main_experiment() {
    local datasets="$1"
    local device="$2"
    local output_dir="$3"
    local resume="$4"
    local dry_run="$5"
    
    log_info "=========================================="
    log_info "Running Main Experiments"
    log_info "=========================================="
    log_info "Datasets: ${datasets}"
    log_info "Device: ${device}"
    log_info "Output: ${output_dir}/main_experiments"
    
    local cmd="python3 experiments/run_main.py"
    cmd+=" --datasets ${datasets}"
    cmd+=" --device ${device}"
    cmd+=" --results_dir ${output_dir}/main_experiments"
    
    if [[ "${resume}" == "true" ]]; then
        cmd+=" --resume"
    fi
    
    log_info "Command: ${cmd}"
    
    if [[ "${dry_run}" != "true" ]]; then
        if eval "${cmd}"; then
            log_success "Main experiments completed"
            return 0
        else
            log_error "Main experiments failed"
            return 1
        fi
    else
        log_info "[DRY RUN] Would execute: ${cmd}"
        return 0
    fi
}

run_ablation_study() {
    local datasets="$1"
    local device="$2"
    local output_dir="$3"
    local dry_run="$4"
    
    log_info "=========================================="
    log_info "Running Ablation Study"
    log_info "=========================================="
    log_info "Datasets: ${datasets}"
    log_info "Device: ${device}"
    log_info "Output: ${output_dir}/ablation"
    
    local cmd="python3 experiments/run_ablation.py"
    cmd+=" --dataset ${datasets%%,*}"  # Use first dataset
    cmd+=" --device ${device}"
    cmd+=" --results_dir ${output_dir}/ablation"
    
    log_info "Command: ${cmd}"
    
    if [[ "${dry_run}" != "true" ]]; then
        if eval "${cmd}"; then
            log_success "Ablation study completed"
            return 0
        else
            log_error "Ablation study failed"
            return 1
        fi
    else
        log_info "[DRY RUN] Would execute: ${cmd}"
        return 0
    fi
}

run_domain_generalization() {
    local device="$1"
    local output_dir="$2"
    local dry_run="$3"
    
    log_info "=========================================="
    log_info "Running Domain Generalization Experiments"
    log_info "=========================================="
    log_info "Device: ${device}"
    log_info "Output: ${output_dir}/domain_generalization"
    
    local cmd="python3 experiments/run_domain_generalization.py"
    cmd+=" --device ${device}"
    cmd+=" --results_dir ${output_dir}/domain_generalization"
    
    log_info "Command: ${cmd}"
    
    if [[ "${dry_run}" != "true" ]]; then
        if eval "${cmd}"; then
            log_success "Domain generalization experiments completed"
            return 0
        else
            log_error "Domain generalization experiments failed"
            return 1
        fi
    else
        log_info "[DRY RUN] Would execute: ${cmd}"
        return 0
    fi
}

run_hyperparameter_search() {
    local datasets="$1"
    local device="$2"
    local output_dir="$3"
    local dry_run="$4"
    
    log_info "=========================================="
    log_info "Running Hyperparameter Search"
    log_info "=========================================="
    log_info "Datasets: ${datasets}"
    log_info "Device: ${device}"
    log_info "Output: ${output_dir}/hyperparameter_search"
    
    local cmd="python3 experiments/run_hyperparameter_search.py"
    cmd+=" --dataset ${datasets%%,*}"  # Use first dataset
    cmd+=" --device ${device}"
    cmd+=" --results_dir ${output_dir}/hyperparameter_search"
    
    log_info "Command: ${cmd}"
    
    if [[ "${dry_run}" != "true" ]]; then
        if eval "${cmd}"; then
            log_success "Hyperparameter search completed"
            return 0
        else
            log_error "Hyperparameter search failed"
            return 1
        fi
    else
        log_info "[DRY RUN] Would execute: ${cmd}"
        return 0
    fi
}

# ==============================================================================
# RESULTS SUMMARY FUNCTIONS
# ==============================================================================
generate_results_summary() {
    local output_dir="$1"
    
    log_info "=========================================="
    log_info "Generating Results Summary"
    log_info "=========================================="
    
    local summary_file="${output_dir}/experiment_summary.json"
    
    python3 << EOF
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("${output_dir}")
summary = {
    'timestamp': datetime.now().isoformat(),
    'experiments': {}
}

# Check each experiment directory
for exp_type in ['main_experiments', 'ablation', 'domain_generalization', 'hyperparameter_search']:
    exp_dir = output_dir / exp_type
    if exp_dir.exists():
        result_files = list(exp_dir.glob('*.json'))
        summary['experiments'][exp_type] = {
            'completed': len(result_files) > 0,
            'num_results': len(result_files),
            'files': [str(f) for f in result_files]
        }

# Save summary
with open('${summary_file}', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: ${summary_file}")
EOF
    
    log_success "Results summary generated"
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
main() {
    # Default values
    local experiment="all"
    local datasets="MNIST,FashionMNIST,CIFAR-10,CIFAR-100,CelebA,TinyImageNet"
    local device="${DEFAULT_DEVICE}"
    local output_dir="${DEFAULT_OUTPUT_DIR}"
    local resume="false"
    local dry_run="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --experiment)
                experiment="$2"
                shift 2
                ;;
            --datasets)
                datasets="$2"
                shift 2
                ;;
            --device)
                device="$2"
                shift 2
                ;;
            --output_dir)
                output_dir="$2"
                shift 2
                ;;
            --resume)
                resume="true"
                shift
                ;;
            --dry_run)
                dry_run="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate experiment type
    if [[ ! ",${EXPERIMENT_TYPES}," =~ ",${experiment}," ]]; then
        log_error "Invalid experiment type: ${experiment}"
        log_info "Valid types: ${EXPERIMENT_TYPES}"
        exit 1
    fi
    
    # Start logging
    log_info "=========================================="
    log_info "${SCRIPT_NAME} v${SCRIPT_VERSION}"
    log_info "=========================================="
    log_info "Experiment: ${experiment}"
    log_info "Datasets: ${datasets}"
    log_info "Device: ${device}"
    log_info "Output: ${output_dir}"
    log_info "Resume: ${resume}"
    log_info "Dry Run: ${dry_run}"
    
    # Check dependencies
    check_dependencies
    
    # Create directories
    create_directories "${output_dir}"
    
    # Track success/failure
    local success_count=0
    local failure_count=0
    
    # Run experiments based on type
    case "${experiment}" in
        "main")
            if run_main_experiment "${datasets}" "${device}" "${output_dir}" "${resume}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "ablation")
            if run_ablation_study "${datasets}" "${device}" "${output_dir}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "domain")
            if run_domain_generalization "${device}" "${output_dir}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "hyperparam")
            if run_hyperparameter_search "${datasets}" "${device}" "${output_dir}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "all")
            # Run all experiments
            if run_main_experiment "${datasets}" "${device}" "${output_dir}" "${resume}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            
            if run_ablation_study "${datasets}" "${device}" "${output_dir}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            
            if run_domain_generalization "${device}" "${output_dir}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            
            if run_hyperparameter_search "${datasets}" "${device}" "${output_dir}" "${dry_run}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
    esac
    
    # Generate summary
    if [[ "${dry_run}" != "true" ]]; then
        generate_results_summary "${output_dir}"
    fi
    
    # Summary
    log_info "=========================================="
    log_info "EXPERIMENT SUMMARY"
    log_info "=========================================="
    log_info "Successful: ${success_count}"
    log_info "Failed: ${failure_count}"
    
    if [[ ${failure_count} -gt 0 ]]; then
        log_error "Some experiments failed"
        exit 1
    fi
    
    log_success "All experiments completed successfully!"
    log_info "Results location: ${output_dir}"
}

# Run main function
main "$@"