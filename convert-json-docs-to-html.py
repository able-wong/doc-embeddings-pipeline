#!/usr/bin/env python3
"""
JSON to HTML Document Converter

Converts existing JSON documents to HTML format using the shared HTML export functionality.
Processes all JSON files in the input directory and creates corresponding HTML files in the output directory.

Usage:
    python convert-json-docs-to-html.py input_folder output_folder [--overwrite]

Examples:
    # Convert all JSON files from documents/json to documents/html
    python convert-json-docs-to-html.py documents/json documents/html
    
    # Convert with overwriting existing files
    python convert-json-docs-to-html.py documents/json documents/html --overwrite
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from html_exporter import convert_json_to_html, generate_filename


def find_json_files(input_dir: Path) -> List[Path]:
    """
    Find all JSON files in the input directory.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        List of JSON file paths
    """
    json_files = []
    for file_path in input_dir.glob("*.json"):
        if file_path.is_file():
            json_files.append(file_path)
    
    return sorted(json_files)


def load_json_document(json_path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Load and validate a JSON document.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Tuple of (json_data, error_message). error_message is empty on success.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Validate required fields
        if not isinstance(json_data, dict):
            return {}, "JSON file does not contain a valid document object"
        
        # Check for required fields - at minimum we need title or original_text
        if not json_data.get('title') and not json_data.get('original_text'):
            return {}, "JSON document missing both 'title' and 'original_text' fields"
        
        return json_data, ""
        
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON format: {e}"
    except Exception as e:
        return {}, f"Error reading file: {e}"


def convert_single_file(json_path: Path, output_dir: Path, overwrite: bool = False) -> Tuple[bool, str]:
    """
    Convert a single JSON file to HTML.
    
    Args:
        json_path: Path to input JSON file
        output_dir: Output directory for HTML file
        overwrite: Whether to overwrite existing HTML files
        
    Returns:
        Tuple of (success, message)
    """
    # Load JSON document
    json_data, error = load_json_document(json_path)
    if error:
        return False, f"Failed to load {json_path.name}: {error}"
    
    # Generate output filename
    title = json_data.get('title', json_path.stem)
    publication_date = json_data.get('publication_date')
    html_filename = generate_filename(title, publication_date, "html")
    html_path = output_dir / html_filename
    
    # Check if file exists and overwrite flag
    if html_path.exists() and not overwrite:
        return False, f"HTML file already exists: {html_path.name} (use --overwrite to replace)"
    
    try:
        # Convert to HTML
        convert_json_to_html(json_data, str(html_path))
        return True, f"Converted {json_path.name} → {html_path.name}"
        
    except Exception as e:
        return False, f"Failed to convert {json_path.name}: {e}"


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert JSON documents to HTML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all JSON files from documents/json to documents/html
  python convert-json-docs-to-html.py documents/json documents/html
  
  # Convert with overwriting existing files
  python convert-json-docs-to-html.py documents/json documents/html --overwrite
  
  # Convert from current directory to html subfolder
  python convert-json-docs-to-html.py . ./html
        """
    )
    
    parser.add_argument('input_folder', type=str, help='Input folder containing JSON files')
    parser.add_argument('output_folder', type=str, help='Output folder for HTML files')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing HTML files')
    
    args = parser.parse_args()
    
    # Validate input and output directories
    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"❌ Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create output directory {output_dir}: {e}")
        sys.exit(1)
    
    # Find JSON files
    json_files = find_json_files(input_dir)
    
    if not json_files:
        print(f"❌ No JSON files found in {input_dir}")
        sys.exit(1)
    
    print(f"🔍 Found {len(json_files)} JSON files in {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    if args.overwrite:
        print("⚠️  Overwrite mode enabled - existing HTML files will be replaced")
    
    print("\n🚀 Starting conversion...")
    
    # Process each JSON file
    successful = 0
    failed = 0
    skipped = 0
    
    for json_path in json_files:
        success, message = convert_single_file(json_path, output_dir, args.overwrite)
        
        if success:
            print(f"✅ {message}")
            successful += 1
        elif "already exists" in message:
            print(f"⚠️  {message}")
            skipped += 1
        else:
            print(f"❌ {message}")
            failed += 1
    
    # Print summary
    print("\n📊 CONVERSION SUMMARY:")
    print(f"✅ Successfully converted: {successful}")
    if skipped > 0:
        print(f"⚠️  Skipped (already exist): {skipped}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    
    print(f"📁 HTML files saved to: {output_dir}")
    
    # Set exit code based on results
    if failed == 0:
        sys.exit(0)  # Success: all files processed or skipped
    elif successful > 0:
        sys.exit(1)  # Partial success: some files failed
    else:
        sys.exit(2)  # Complete failure: no files converted


if __name__ == "__main__":
    main()