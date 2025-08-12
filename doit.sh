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

# Run tests
run_tests() {
    echo "🧪 Running tests..."
    python -m pytest tests/ -v
    echo "✅ Tests completed successfully!"
}

# Run linter (check only)
run_lint() {
    echo "🔍 Running ruff linter..."
    ruff check .
    echo "✅ Linting completed successfully!"
}

# Run linter with auto-fix
run_lint_fix() {
    echo "🔧 Running ruff linter with auto-fix..."
    ruff check --fix .
    echo "✅ Linting and auto-fix completed successfully!"
}

# Show usage
show_usage() {
    echo "Development script for common tasks"
    echo ""
    echo "Usage: ./doit.sh [command]"
    echo ""
    echo "Commands:"
    echo "  test      Run all tests with pytest"
    echo "  lint      Run ruff linter (check only)"
    echo "  lint-fix  Run ruff linter with auto-fix"
    echo ""
    echo "Examples:"
    echo "  ./doit.sh test"
    echo "  ./doit.sh lint"
    echo "  ./doit.sh lint-fix"
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