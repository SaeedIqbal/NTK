#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Plot Generation Script for NTK-SURGERY

This script generates all visualization plots from experiment results:
- Efficacy plots (Forget/Retain Accuracy, Exactness Score)
- Efficiency plots (Communication Rounds, Server Time)
- Theoretical plots (NTK Alignment, Sensitivity Bounds)
- Ablation plots (Component Contributions)

All plots follow ACCV publication standards.

Usage:
    ./generate_plots.sh [OPTIONS]

Options:
    --results_dir   Directory containing experiment results
    --output_dir    Output directory for plots
    --plot_type     Type of plots to generate (efficacy|efficiency|theoretical|ablation|all)
    --format        Plot format (pdf|png|svg)
    --dpi           DPI for saved plots
    --help          Show this help message

Example:
    ./generate_plots.sh --results_dir results --output_dir figures
    ./generate_plots.sh --plot_type efficacy --format pdf
"""

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
readonly SCRIPT_NAME="generate_plots.sh"
readonly SCRIPT_VERSION="1.0.0"
readonly DEFAULT_RESULTS_DIR="results"
readonly DEFAULT_OUTPUT_DIR="results/visualizations"
readonly DEFAULT_FORMAT="pdf"
readonly DEFAULT_DPI="600"
readonly LOG_FILE="logs/plot_generation.log"

# Plot types
readonly PLOT_TYPES="efficacy,efficiency,theoretical,ablation,all"

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

Generate publication-quality plots from NTK-SURGERY experiment results.

Usage:
    ${SCRIPT_NAME} [OPTIONS]

Options:
    --results_dir   Directory containing experiment results (default: ${DEFAULT_RESULTS_DIR})
    --output_dir    Output directory for plots (default: ${DEFAULT_OUTPUT_DIR})
    --plot_type     Plot type: ${PLOT_TYPES} (default: all)
    --format        Plot format: pdf|png|svg (default: ${DEFAULT_FORMAT})
    --dpi           DPI for saved plots (default: ${DEFAULT_DPI})
    --help          Show this help message

Examples:
    ${SCRIPT_NAME}
    ${SCRIPT_NAME} --plot_type efficacy --format pdf
    ${SCRIPT_NAME} --results_dir /results --output_dir /figures --dpi 300

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
    if ! python3 -c "import matplotlib" &> /dev/null; then
        missing_deps+=("matplotlib (pip install matplotlib)")
    fi
    
    if ! python3 -c "import seaborn" &> /dev/null; then
        missing_deps+=("seaborn (pip install seaborn)")
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
    mkdir -p "${output_dir}/efficacy"
    mkdir -p "${output_dir}/efficiency"
    mkdir -p "${output_dir}/theoretical"
    mkdir -p "${output_dir}/ablation"
    mkdir -p "$(dirname "${LOG_FILE}")"
    mkdir -p "logs"
    
    log_success "Directories created"
}

# ==============================================================================
# PLOT GENERATION FUNCTIONS
# ==============================================================================
find_result_files() {
    local results_dir="$1"
    local experiment_type="$2"
    
    local result_files=()
    
    case "${experiment_type}" in
        "main")
            while IFS= read -r -d '' file; do
                result_files+=("${file}")
            done < <(find "${results_dir}/main_experiments" -name "*.json" -print0 2>/dev/null)
            ;;
        "ablation")
            while IFS= read -r -d '' file; do
                result_files+=("${file}")
            done < <(find "${results_dir}/ablation" -name "*.json" -print0 2>/dev/null)
            ;;
        "domain")
            while IFS= read -r -d '' file; do
                result_files+=("${file}")
            done < <(find "${results_dir}/domain_generalization" -name "*.json" -print0 2>/dev/null)
            ;;
        "hyperparam")
            while IFS= read -r -d '' file; do
                result_files+=("${file}")
            done < <(find "${results_dir}/hyperparameter_search" -name "*.json" -print0 2>/dev/null)
            ;;
    esac
    
    printf '%s\n' "${result_files[@]}"
}

generate_efficacy_plots() {
    local results_dir="$1"
    local output_dir="$2"
    local format="$3"
    local dpi="$4"
    
    log_info "=========================================="
    log_info "Generating Efficacy Plots"
    log_info "=========================================="
    
    # Find result files
    local result_files
    result_files=$(find_result_files "${results_dir}" "main")
    
    if [[ -z "${result_files}" ]]; then
        log_warning "No result files found for efficacy plots"
        return 0
    fi
    
    # Get first result file for plotting
    local first_file
    first_file=$(echo "${result_files}" | head -n 1)
    
    log_info "Using result file: ${first_file}"
    
    python3 << EOF
import sys
sys.path.insert(0, '.')

from visualization.plot_efficacy import plot_unlearning_efficacy, create_efficacy_summary
from pathlib import Path

results_file = "${first_file}"
output_dir = Path("${output_dir}/efficacy")
format = "${format}"
dpi = ${dpi}

try:
    # Generate plots
    plot_paths = plot_unlearning_efficacy(
        results_file=results_file,
        output_dir=str(output_dir),
        config=None
    )
    
    print(f"Generated {len(plot_paths)} efficacy plots")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")
    
    # Create summary
    import json
    with open(results_file, 'r') as f:
        results = json.load(f).get('results', {})
    
    summary_file = output_dir / 'efficacy_summary.json'
    create_efficacy_summary(results, str(summary_file))
    print(f"Summary saved to: {summary_file}")
    
except Exception as e:
    print(f"Error generating efficacy plots: {e}")
    sys.exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Efficacy plots generated"
        return 0
    else
        log_error "Failed to generate efficacy plots"
        return 1
    fi
}

generate_efficiency_plots() {
    local results_dir="$1"
    local output_dir="$2"
    local format="$3"
    local dpi="$4"
    
    log_info "=========================================="
    log_info "Generating Efficiency Plots"
    log_info "=========================================="
    
    # Find result files
    local result_files
    result_files=$(find_result_files "${results_dir}" "main")
    
    if [[ -z "${result_files}" ]]; then
        log_warning "No result files found for efficiency plots"
        return 0
    fi
    
    # Get first result file for plotting
    local first_file
    first_file=$(echo "${result_files}" | head -n 1)
    
    log_info "Using result file: ${first_file}"
    
    python3 << EOF
import sys
sys.path.insert(0, '.')

from visualization.plot_efficiency import plot_efficiency_metrics, create_efficiency_comparison
from pathlib import Path

results_file = "${first_file}"
output_dir = Path("${output_dir}/efficiency")
format = "${format}"
dpi = ${dpi}

try:
    # Generate plots
    plot_paths = plot_efficiency_metrics(
        results_file=results_file,
        output_dir=str(output_dir),
        config=None
    )
    
    print(f"Generated {len(plot_paths)} efficiency plots")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")
    
    # Create summary
    import json
    with open(results_file, 'r') as f:
        results = json.load(f).get('results', {})
    
    summary_file = output_dir / 'efficiency_summary.json'
    create_efficiency_comparison(results, str(summary_file))
    print(f"Summary saved to: {summary_file}")
    
except Exception as e:
    print(f"Error generating efficiency plots: {e}")
    sys.exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Efficiency plots generated"
        return 0
    else
        log_error "Failed to generate efficiency plots"
        return 1
    fi
}

generate_theoretical_plots() {
    local results_dir="$1"
    local output_dir="$2"
    local format="$3"
    local dpi="$4"
    
    log_info "=========================================="
    log_info "Generating Theoretical Plots"
    log_info "=========================================="
    
    # Find result files
    local result_files
    result_files=$(find_result_files "${results_dir}" "main")
    
    if [[ -z "${result_files}" ]]; then
        log_warning "No result files found for theoretical plots"
        return 0
    fi
    
    # Get first result file for plotting
    local first_file
    first_file=$(echo "${result_files}" | head -n 1)
    
    log_info "Using result file: ${first_file}"
    
    python3 << EOF
import sys
sys.path.insert(0, '.')

from visualization.plot_theoretical import plot_theoretical_metrics, create_theoretical_analysis
from pathlib import Path

results_file = "${first_file}"
output_dir = Path("${output_dir}/theoretical")
format = "${format}"
dpi = ${dpi}

try:
    # Generate plots
    plot_paths = plot_theoretical_metrics(
        results_file=results_file,
        output_dir=str(output_dir),
        config=None
    )
    
    print(f"Generated {len(plot_paths)} theoretical plots")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")
    
    # Create summary
    import json
    with open(results_file, 'r') as f:
        results = json.load(f).get('results', {})
    
    summary_file = output_dir / 'theoretical_summary.json'
    create_theoretical_analysis(results, str(summary_file))
    print(f"Summary saved to: {summary_file}")
    
except Exception as e:
    print(f"Error generating theoretical plots: {e}")
    sys.exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Theoretical plots generated"
        return 0
    else
        log_error "Failed to generate theoretical plots"
        return 1
    fi
}

generate_ablation_plots() {
    local results_dir="$1"
    local output_dir="$2"
    local format="$3"
    local dpi="$4"
    
    log_info "=========================================="
    log_info "Generating Ablation Plots"
    log_info "=========================================="
    
    # Find result files
    local result_files
    result_files=$(find_result_files "${results_dir}" "ablation")
    
    if [[ -z "${result_files}" ]]; then
        log_warning "No result files found for ablation plots"
        return 0
    fi
    
    # Get first result file for plotting
    local first_file
    first_file=$(echo "${result_files}" | head -n 1)
    
    log_info "Using result file: ${first_file}"
    
    python3 << EOF
import sys
sys.path.insert(0, '.')

from visualization.plot_ablation import plot_ablation_study, create_ablation_summary
from pathlib import Path

results_file = "${first_file}"
output_dir = Path("${output_dir}/ablation")
format = "${format}"
dpi = ${dpi}

try:
    # Generate plots
    plot_paths = plot_ablation_study(
        results_file=results_file,
        output_dir=str(output_dir),
        config=None
    )
    
    print(f"Generated {len(plot_paths)} ablation plots")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")
    
    # Create summary
    import json
    with open(results_file, 'r') as f:
        results = json.load(f).get('results', {})
    
    summary_file = output_dir / 'ablation_summary.json'
    create_ablation_summary(results, str(summary_file))
    print(f"Summary saved to: {summary_file}")
    
except Exception as e:
    print(f"Error generating ablation plots: {e}")
    sys.exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Ablation plots generated"
        return 0
    else
        log_error "Failed to generate ablation plots"
        return 1
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
main() {
    # Default values
    local results_dir="${DEFAULT_RESULTS_DIR}"
    local output_dir="${DEFAULT_OUTPUT_DIR}"
    local plot_type="all"
    local format="${DEFAULT_FORMAT}"
    local dpi="${DEFAULT_DPI}"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --results_dir)
                results_dir="$2"
                shift 2
                ;;
            --output_dir)
                output_dir="$2"
                shift 2
                ;;
            --plot_type)
                plot_type="$2"
                shift 2
                ;;
            --format)
                format="$2"
                shift 2
                ;;
            --dpi)
                dpi="$2"
                shift 2
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
    
    # Validate plot type
    if [[ ! ",${PLOT_TYPES}," =~ ",${plot_type}," ]]; then
        log_error "Invalid plot type: ${plot_type}"
        log_info "Valid types: ${PLOT_TYPES}"
        exit 1
    fi
    
    # Validate results directory
    if [[ ! -d "${results_dir}" ]]; then
        log_error "Results directory not found: ${results_dir}"
        log_info "Please run experiments first: ./run_experiments.sh"
        exit 1
    fi
    
    # Start logging
    log_info "=========================================="
    log_info "${SCRIPT_NAME} v${SCRIPT_VERSION}"
    log_info "=========================================="
    log_info "Results Dir: ${results_dir}"
    log_info "Output Dir: ${output_dir}"
    log_info "Plot Type: ${plot_type}"
    log_info "Format: ${format}"
    log_info "DPI: ${dpi}"
    
    # Check dependencies
    check_dependencies
    
    # Create directories
    create_directories "${output_dir}"
    
    # Track success/failure
    local success_count=0
    local failure_count=0
    
    # Generate plots based on type
    case "${plot_type}" in
        "efficacy")
            if generate_efficacy_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "efficiency")
            if generate_efficiency_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "theoretical")
            if generate_theoretical_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "ablation")
            if generate_ablation_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
        "all")
            # Generate all plots
            if generate_efficacy_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            
            if generate_efficiency_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            
            if generate_theoretical_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            
            if generate_ablation_plots "${results_dir}" "${output_dir}" "${format}" "${dpi}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
            ;;
    esac
    
    # Summary
    log_info "=========================================="
    log_info "PLOT GENERATION SUMMARY"
    log_info "=========================================="
    log_info "Successful: ${success_count}"
    log_info "Failed: ${failure_count}"
    log_info "Output location: ${output_dir}"
    
    if [[ ${failure_count} -gt 0 ]]; then
        log_error "Some plot generation failed"
        exit 1
    fi
    
    log_success "All plots generated successfully!"
    
    # List generated files
    log_info "Generated plot files:"
    find "${output_dir}" -name "*.${format}" -type f | while read -r file; do
        log_info "  - ${file}"
    done
}

# Run main function
main "$@"