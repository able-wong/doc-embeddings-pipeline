#!/bin/bash

# Development script for common tasks
# Usage: ./doit.sh [command]
# Commands: test, lint, lint-fix

set -e

# Check if we're in a virtual environment
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "❌ Error: Virtual environment not activated!"
        echo ""
        echo "Please activate your virtual environment first:"
        echo "  source venv/bin/activate"
        echo ""
        echo "Or create one if it doesn't exist:"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
    
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
}

# Run unit tests (excludes integration tests)
run_tests() {
    echo "🧪 Running unit tests..."
    python -m pytest tests/ --ignore=tests/integration -v
    echo "✅ Unit tests completed successfully!"
}

# Run integration tests (real API calls)
run_integration_tests() {
    echo "🔗 Running integration tests..."
    if [ ! -d "tests/integration" ]; then
        echo "❌ Integration tests directory not found"
        echo "Integration tests require real API keys and running services."
        echo "See tests/integration/README.md for setup instructions."
        exit 1
    fi
    python -m pytest tests/integration/ -v
    echo "✅ Integration tests completed successfully!"
}

# Run all tests (unit + integration)
run_all_tests() {
    echo "🧪 Running all tests (unit + integration)..."
    python -m pytest tests/ -v
    echo "✅ All tests completed successfully!"
}

# Run linter (check only)
run_lint() {
    echo "🔍 Running ruff linter and format check..."
    ruff check . && ruff format --check .
    echo "✅ Linting and formatting checks completed successfully!"
}

# Run linter with auto-fix
run_lint_fix() {
    echo "🔧 Running ruff linter with auto-fix and formatting..."
    ruff check --fix . && ruff format .
    echo "✅ Linting, auto-fix, and formatting completed successfully!"
}

# Show usage
show_usage() {
    echo "Development script for common tasks"
    echo ""
    echo "Usage: ./doit.sh [command]"
    echo ""
    echo "Commands:"
    echo "  test             Run unit tests only (default, matches CI)"
    echo "  test-integration Run integration tests (requires real APIs)"
    echo "  test-all         Run all tests (unit + integration)"
    echo "  lint             Run ruff linter and format check (matches CI)"
    echo "  lint-fix         Run ruff linter with auto-fix and formatting"
    echo ""
    echo "Examples:"
    echo "  ./doit.sh test               # Unit tests only"
    echo "  ./doit.sh test-integration   # Integration tests only"
    echo "  ./doit.sh test-all           # All tests"
    echo "  ./doit.sh lint               # Check linting"
    echo "  ./doit.sh lint-fix           # Fix linting issues"
    echo ""
    echo "Note: Virtual environment must be activated before running commands."
}

# Main script logic
main() {
    # Always check venv first
    check_venv
    
    case "${1:-}" in
        "test")
            run_tests
            ;;
        "test-integration")
            run_integration_tests
            ;;
        "test-all")
            run_all_tests
            ;;
        "lint")
            run_lint
            ;;
        "lint-fix")
            run_lint_fix
            ;;
        "")
            show_usage
            ;;
        *)
            echo "❌ Error: Unknown command '$1'"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
