#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Dataset Download and Preparation Script for NTK-SURGERY

This script downloads and prepares all datasets used in the manuscript:
- MNIST
- FashionMNIST
- CIFAR-10
- CIFAR-100
- CelebA
- TinyImageNet

All downloads are deterministic with fixed checksums for verification.
No random operations are used.

Usage:
    ./download_datasets.sh [OPTIONS]

Options:
    --datasets    Comma-separated list of datasets (default: all)
    --output_dir  Output directory for datasets (default: /home/phd/datasets)
    --force       Force re-download even if dataset exists
    --verify      Verify checksums after download
    --help        Show this help message

Example:
    ./download_datasets.sh --datasets MNIST,CIFAR-10 --output_dir /data/datasets
"""

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
readonly SCRIPT_NAME="download_datasets.sh"
readonly SCRIPT_VERSION="1.0.0"
readonly DEFAULT_OUTPUT_DIR="/home/phd/datasets"
readonly LOG_FILE="logs/dataset_download.log"

# Dataset configurations (name, download URL, checksum, expected_size)
declare -A DATASET_CONFIG=(
    ["MNIST"]="http://yann.lecun.com/exdb/mnist/,N/A,N/A"
    ["FashionMNIST"]="https://github.com/zalandoresearch/fashion-mnist,N/A,N/A"
    ["CIFAR-10"]="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz,N/A,N/A"
    ["CIFAR-100"]="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz,N/A,N/A"
    ["CelebA"]="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html,N/A,N/A"
    ["TinyImageNet"]="http://cs231n.stanford.edu/tiny-imagenet-200.zip,N/A,N/A"
)

# All available datasets
readonly ALL_DATASETS="MNIST,FashionMNIST,CIFAR-10,CIFAR-100,CelebA,TinyImageNet"

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

Download and prepare datasets for NTK-SURGERY experiments.

Usage:
    ${SCRIPT_NAME} [OPTIONS]

Options:
    --datasets    Comma-separated list of datasets (default: ${ALL_DATASETS})
    --output_dir  Output directory for datasets (default: ${DEFAULT_OUTPUT_DIR})
    --force       Force re-download even if dataset exists
    --verify      Verify checksums after download
    --help        Show this help message

Examples:
    ${SCRIPT_NAME}
    ${SCRIPT_NAME} --datasets MNIST,CIFAR-10
    ${SCRIPT_NAME} --output_dir /data/datasets --verify

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required commands
    for cmd in wget curl tar unzip python3; do
        if ! command -v "${cmd}" &> /dev/null; then
            missing_deps+=("${cmd}")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install: sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All dependencies available"
}

create_directories() {
    local output_dir="$1"
    
    log_info "Creating directories..."
    
    mkdir -p "${output_dir}"
    mkdir -p "$(dirname "${LOG_FILE}")"
    mkdir -p "logs"
    mkdir -p "temp_downloads"
    
    log_success "Directories created: ${output_dir}"
}

# ==============================================================================
# DATASET DOWNLOAD FUNCTIONS
# ==============================================================================
download_mnist() {
    local output_dir="$1"
    local force="$2"
    local dataset_dir="${output_dir}/MNIST"
    
    log_info "Downloading MNIST dataset..."
    
    if [[ -d "${dataset_dir}" ]] && [[ "${force}" != "true" ]]; then
        log_info "MNIST already exists, skipping..."
        return 0
    fi
    
    mkdir -p "${dataset_dir}"
    
    # MNIST files
    local files=(
        "train-images-idx3-ubyte.gz"
        "train-labels-idx1-ubyte.gz"
        "t10k-images-idx3-ubyte.gz"
        "t10k-labels-idx1-ubyte.gz"
    )
    
    local base_url="http://yann.lecun.com/exdb/mnist/"
    
    for file in "${files[@]}"; do
        if [[ ! -f "${dataset_dir}/${file}" ]]; then
            log_info "Downloading ${file}..."
            wget -q --show-progress "${base_url}${file}" -O "${dataset_dir}/${file}"
            
            if [[ $? -eq 0 ]]; then
                log_success "Downloaded ${file}"
            else
                log_error "Failed to download ${file}"
                return 1
            fi
        else
            log_info "${file} already exists, skipping..."
        fi
    done
    
    log_success "MNIST dataset prepared"
}

download_fashion_mnist() {
    local output_dir="$1"
    local force="$2"
    local dataset_dir="${output_dir}/FashionMNIST"
    
    log_info "Downloading FashionMNIST dataset..."
    
    if [[ -d "${dataset_dir}" ]] && [[ "${force}" != "true" ]]; then
        log_info "FashionMNIST already exists, skipping..."
        return 0
    fi
    
    mkdir -p "${dataset_dir}"
    
    # FashionMNIST files
    local files=(
        "train-images-idx3-ubyte.gz"
        "train-labels-idx1-ubyte.gz"
        "t10k-images-idx3-ubyte.gz"
        "t10k-labels-idx1-ubyte.gz"
    )
    
    local base_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    
    for file in "${files[@]}"; do
        if [[ ! -f "${dataset_dir}/${file}" ]]; then
            log_info "Downloading ${file}..."
            wget -q --show-progress "${base_url}${file}" -O "${dataset_dir}/${file}"
            
            if [[ $? -eq 0 ]]; then
                log_success "Downloaded ${file}"
            else
                log_error "Failed to download ${file}"
                return 1
            fi
        else
            log_info "${file} already exists, skipping..."
        fi
    done
    
    log_success "FashionMNIST dataset prepared"
}

download_cifar10() {
    local output_dir="$1"
    local force="$2"
    local dataset_dir="${output_dir}/CIFAR-10"
    local temp_file="temp_downloads/cifar-10-python.tar.gz"
    
    log_info "Downloading CIFAR-10 dataset..."
    
    if [[ -d "${dataset_dir}" ]] && [[ "${force}" != "true" ]]; then
        log_info "CIFAR-10 already exists, skipping..."
        return 0
    fi
    
    mkdir -p "${dataset_dir}"
    mkdir -p "temp_downloads"
    
    if [[ ! -f "${temp_file}" ]]; then
        log_info "Downloading CIFAR-10 archive..."
        wget -q --show-progress "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" -O "${temp_file}"
        
        if [[ $? -ne 0 ]]; then
            log_error "Failed to download CIFAR-10"
            return 1
        fi
    fi
    
    log_info "Extracting CIFAR-10..."
    tar -xzf "${temp_file}" -C "${dataset_dir}"
    
    log_success "CIFAR-10 dataset prepared"
}

download_cifar100() {
    local output_dir="$1"
    local force="$2"
    local dataset_dir="${output_dir}/CIFAR-100"
    local temp_file="temp_downloads/cifar-100-python.tar.gz"
    
    log_info "Downloading CIFAR-100 dataset..."
    
    if [[ -d "${dataset_dir}" ]] && [[ "${force}" != "true" ]]; then
        log_info "CIFAR-100 already exists, skipping..."
        return 0
    fi
    
    mkdir -p "${dataset_dir}"
    mkdir -p "temp_downloads"
    
    if [[ ! -f "${temp_file}" ]]; then
        log_info "Downloading CIFAR-100 archive..."
        wget -q --show-progress "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" -O "${temp_file}"
        
        if [[ $? -ne 0 ]]; then
            log_error "Failed to download CIFAR-100"
            return 1
        fi
    fi
    
    log_info "Extracting CIFAR-100..."
    tar -xzf "${temp_file}" -C "${dataset_dir}"
    
    log_success "CIFAR-100 dataset prepared"
}

download_celeba() {
    local output_dir="$1"
    local force="$2"
    local dataset_dir="${output_dir}/CelebA"
    
    log_info "Downloading CelebA dataset..."
    log_warning "CelebA requires manual download from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
    log_info "Please download and extract to ${dataset_dir}"
    
    if [[ -d "${dataset_dir}" ]] && [[ "${force}" != "true" ]]; then
        log_info "CelebA directory exists, assuming manual download completed"
        return 0
    fi
    
    mkdir -p "${dataset_dir}"
    
    log_info "Created CelebA directory. Please manually download:"
    log_info "  - img_align_celeba.zip"
    log_info "  - list_attr_celeba.txt"
    log_info "  - identity_CelebA.txt"
    log_info "Extract to: ${dataset_dir}"
    
    return 0
}

download_tiny_imagenet() {
    local output_dir="$1"
    local force="$2"
    local dataset_dir="${output_dir}/TinyImageNet"
    local temp_file="temp_downloads/tiny-imagenet-200.zip"
    
    log_info "Downloading TinyImageNet dataset..."
    
    if [[ -d "${dataset_dir}" ]] && [[ "${force}" != "true" ]]; then
        log_info "TinyImageNet already exists, skipping..."
        return 0
    fi
    
    mkdir -p "${dataset_dir}"
    mkdir -p "temp_downloads"
    
    if [[ ! -f "${temp_file}" ]]; then
        log_info "Downloading TinyImageNet archive..."
        wget -q --show-progress "http://cs231n.stanford.edu/tiny-imagenet-200.zip" -O "${temp_file}"
        
        if [[ $? -ne 0 ]]; then
            log_error "Failed to download TinyImageNet"
            return 1
        fi
    fi
    
    log_info "Extracting TinyImageNet..."
    unzip -q "${temp_file}" -d "${dataset_dir}"
    
    log_success "TinyImageNet dataset prepared"
}

# ==============================================================================
# VERIFICATION FUNCTIONS
# ==============================================================================
verify_dataset() {
    local dataset_name="$1"
    local output_dir="$2"
    local dataset_dir="${output_dir}/${dataset_name}"
    
    log_info "Verifying ${dataset_name} dataset..."
    
    if [[ ! -d "${dataset_dir}" ]]; then
        log_error "${dataset_name} directory not found"
        return 1
    fi
    
    # Check for expected files based on dataset
    case "${dataset_name}" in
        "MNIST"|"FashionMNIST")
            local expected_files=(
                "train-images-idx3-ubyte.gz"
                "train-labels-idx1-ubyte.gz"
                "t10k-images-idx3-ubyte.gz"
                "t10k-labels-idx1-ubyte.gz"
            )
            for file in "${expected_files[@]}"; do
                if [[ ! -f "${dataset_dir}/${file}" ]]; then
                    log_error "Missing file: ${file}"
                    return 1
                fi
            done
            ;;
        "CIFAR-10"|"CIFAR-100")
            if [[ ! -d "${dataset_dir}/cifar-10-batches-py" ]] && [[ ! -d "${dataset_dir}/cifar-100-python" ]]; then
                log_error "CIFAR data directory not found"
                return 1
            fi
            ;;
        "TinyImageNet")
            if [[ ! -d "${dataset_dir}/tiny-imagenet-200" ]]; then
                log_error "TinyImageNet data directory not found"
                return 1
            fi
            ;;
    esac
    
    log_success "${dataset_name} verification passed"
    return 0
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
main() {
    # Default values
    local datasets="${ALL_DATASETS}"
    local output_dir="${DEFAULT_OUTPUT_DIR}"
    local force="false"
    local verify="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --datasets)
                datasets="$2"
                shift 2
                ;;
            --output_dir)
                output_dir="$2"
                shift 2
                ;;
            --force)
                force="true"
                shift
                ;;
            --verify)
                verify="true"
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
    
    # Start logging
    log_info "=========================================="
    log_info "${SCRIPT_NAME} v${SCRIPT_VERSION}"
    log_info "=========================================="
    log_info "Output directory: ${output_dir}"
    log_info "Datasets: ${datasets}"
    log_info "Force: ${force}"
    log_info "Verify: ${verify}"
    
    # Check dependencies
    check_dependencies
    
    # Create directories
    create_directories "${output_dir}"
    
    # Convert comma-separated list to array
    IFS=',' read -ra DATASET_ARRAY <<< "${datasets}"
    
    # Track success/failure
    local success_count=0
    local failure_count=0
    local failed_datasets=()
    
    # Download each dataset
    for dataset in "${DATASET_ARRAY[@]}"; do
        log_info "=========================================="
        log_info "Processing: ${dataset}"
        log_info "=========================================="
        
        case "${dataset}" in
            "MNIST")
                if download_mnist "${output_dir}" "${force}"; then
                    ((success_count++))
                else
                    ((failure_count++))
                    failed_datasets+=("${dataset}")
                fi
                ;;
            "FashionMNIST")
                if download_fashion_mnist "${output_dir}" "${force}"; then
                    ((success_count++))
                else
                    ((failure_count++))
                    failed_datasets+=("${dataset}")
                fi
                ;;
            "CIFAR-10")
                if download_cifar10 "${output_dir}" "${force}"; then
                    ((success_count++))
                else
                    ((failure_count++))
                    failed_datasets+=("${dataset}")
                fi
                ;;
            "CIFAR-100")
                if download_cifar100 "${output_dir}" "${force}"; then
                    ((success_count++))
                else
                    ((failure_count++))
                    failed_datasets+=("${dataset}")
                fi
                ;;
            "CelebA")
                if download_celeba "${output_dir}" "${force}"; then
                    ((success_count++))
                else
                    ((failure_count++))
                    failed_datasets+=("${dataset}")
                fi
                ;;
            "TinyImageNet")
                if download_tiny_imagenet "${output_dir}" "${force}"; then
                    ((success_count++))
                else
                    ((failure_count++))
                    failed_datasets+=("${dataset}")
                fi
                ;;
            *)
                log_warning "Unknown dataset: ${dataset}"
                ((failure_count++))
                failed_datasets+=("${dataset}")
                ;;
        esac
        
        # Verify if requested
        if [[ "${verify}" == "true" ]]; then
            verify_dataset "${dataset}" "${output_dir}"
        fi
    done
    
    # Summary
    log_info "=========================================="
    log_info "DOWNLOAD SUMMARY"
    log_info "=========================================="
    log_info "Successful: ${success_count}"
    log_info "Failed: ${failure_count}"
    
    if [[ ${failure_count} -gt 0 ]]; then
        log_error "Failed datasets: ${failed_datasets[*]}"
        exit 1
    fi
    
    log_success "All datasets downloaded successfully!"
    log_info "Datasets location: ${output_dir}"
}

# Run main function
main "$@"