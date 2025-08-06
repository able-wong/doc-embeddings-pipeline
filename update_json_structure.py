#!/usr/bin/env python3
"""
Script to update JSON file structure to match Qdrant collection schema.
Maps fields and validates/regenerates tags as needed.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def validate_tags(tags: List[str], original_text: str) -> List[str]:
    """Validate tags and regenerate if they seem invalid."""
    if not tags or not isinstance(tags, list):
        return generate_tags_from_text(original_text)
    
    # Check if tags seem valid
    valid_tags = []
    for tag in tags:
        if isinstance(tag, str) and len(tag.strip()) > 0 and len(tag) < 50:
            # Remove obviously invalid tags
            tag = tag.strip().lower()
            if not any(invalid in tag for invalid in ['http', 'www', 'com', '.', '@']):
                valid_tags.append(tag)
    
    # If we have few valid tags, regenerate
    if len(valid_tags) < 2:
        return generate_tags_from_text(original_text)
    
    return valid_tags[:8]  # Limit to 8 tags


def generate_tags_from_text(text: str) -> List[str]:
    """Generate tags from text content using simple keyword extraction."""
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Common AI/tech keywords to look for
    keyword_patterns = {
        'artificial intelligence': ['ai', 'artificial intelligence'],
        'machine learning': ['machine learning', 'ml'],
        'large language model': ['llm', 'large language model'],
        'prompt engineering': ['prompt engineering', 'prompting'],
        'automation': ['automation', 'automated'],
        'business': ['business', 'enterprise', 'company'],
        'development': ['development', 'coding', 'programming'],
        'productivity': ['productivity', 'efficiency'],
        'content creation': ['content creation', 'content'],
        'technology': ['technology', 'tech'],
        'innovation': ['innovation', 'innovative'],
        'digital transformation': ['digital transformation', 'digitization'],
        'data': ['data', 'analytics'],
        'cloud': ['cloud', 'aws', 'azure', 'gcp'],
        'devops': ['devops', 'deployment'],
        'security': ['security', 'cybersecurity'],
        'ethics': ['ethics', 'ethical', 'responsible'],
        'future': ['future', 'trends'],
        'workplace': ['workplace', 'work'],
        'research': ['research', 'study']
    }
    
    found_tags = []
    for tag, patterns in keyword_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                found_tags.append(tag)
                break
    
    # Remove duplicates and limit
    unique_tags = list(dict.fromkeys(found_tags))
    return unique_tags[:6]


def convert_date_to_iso(date_str: str) -> str:
    """Convert YYYY-MM-DD to ISO datetime format."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.isoformat()
    except ValueError:
        # If parsing fails, return current timestamp
        return datetime.now().isoformat()


def update_json_structure(file_path: Path) -> Dict[str, Any]:
    """Update a single JSON file structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate tags or regenerate
    original_text = data.get('summary', '')
    tags = validate_tags(data.get('tags', []), original_text)
    
    # Create new structure
    updated_data = {
        'title': data.get('title', ''),
        'author': data.get('author', ''),
        'publication_date': convert_date_to_iso(data.get('publish_date', '')),
        'original_text': original_text,
        'source_url': data.get('original_url', ''),
        'notes': data.get('evaluation_notes'),  # Keep as null if null
        'tags': tags
    }
    
    return updated_data


def main():
    """Process all JSON files in documents/json/ directory."""
    json_dir = Path('documents/json')
    
    if not json_dir.exists():
        print(f"Directory {json_dir} not found!")
        return
    
    json_files = list(json_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files to process")
    
    for file_path in json_files:
        try:
            print(f"Processing {file_path.name}...")
            updated_data = update_json_structure(file_path)
            
            # Write back to same file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Updated {file_path.name}")
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    print(f"\n🎉 Processed {len(json_files)} files successfully!")


if __name__ == '__main__':
    main()