#!/usr/bin/env python3
"""
Simple test of the deep merge functionality without external dependencies.
"""

def deep_merge_settings(existing_settings, new_settings):
    """
    Deep merge new settings into existing settings, preserving all existing fields
    and only updating the fields that are explicitly provided in new_settings.
    """
    if not existing_settings:
        return new_settings.copy()
    
    merged = existing_settings.copy()
    
    for key, value in new_settings.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = deep_merge_settings(merged[key], value)
        else:
            # Override with new value
            merged[key] = value
    
    return merged

def test_deep_merge():
    """Test the deep merge functionality"""
    print("=== Testing Deep Merge Functionality ===")
    
    # Simulate existing complex database settings
    existing_settings = {
        'model': 'old_model',
        'model_config': {
            'model': 'old_model',
            'temperature': 0.2,
            'existing_field': 'preserve_me',
            'nested_config': {
                'deep_setting': 'keep_this'
            }
        },
        'prompts': [
            {'name': 'extraction', 'content': 'complex extraction prompt'},
            {'name': 'discovery', 'content': 'complex discovery prompt'}
        ],
        'extraction': {
            'min_entity_confidence': 0.8,
            'enable_complex_feature': True,
            'complex_nested': {
                'deep_setting': 'important_value',
                'another_deep': {'very_deep': 'critical_data'}
            }
        },
        'learning': {
            'enable_user_feedback': True,
            'learning_data': {'model_performance': [1, 2, 3]}
        },
        'discovered_schemas': {
            'entities': {'Person': {'count': 100}, 'Company': {'count': 50}},
            'relationships': {'WORKS_AT': {'count': 75}},
            'version': '2.0.0'
        },
        'neo4j': {
            'host': 'existing_host',
            'port': 7687,
            'complex_neo4j_setting': 'preserve_this'
        }
    }
    
    # Simulate new settings from frontend (only 3 fields)
    new_frontend_settings = {
        'model_config': {
            'model': 'new_model',
            'temperature': 0.1
            # Note: existing_field and nested_config should be preserved
        },
        'neo4j': {
            'host': 'new_host',
            'port': 7687
            # Note: complex_neo4j_setting should be preserved
        },
        'anti_silo': {
            'enabled': True,
            'similarity_threshold': 0.5
        }
    }
    
    print(f"Existing settings keys: {list(existing_settings.keys())}")
    print(f"New frontend settings keys: {list(new_frontend_settings.keys())}")
    
    # Test deep merge
    merged = deep_merge_settings(existing_settings, new_frontend_settings)
    
    print(f"Merged settings keys: {list(merged.keys())}")
    
    # Verify critical fields are preserved
    critical_fields = ['prompts', 'extraction', 'learning', 'discovered_schemas']
    preserved = [field for field in critical_fields if field in merged and merged[field]]
    print(f"Preserved critical fields: {preserved}")
    
    # Test assertions
    errors = []
    
    # Verify updates worked
    if merged['model_config']['model'] != 'new_model':
        errors.append("Model should be updated")
    if merged['model_config']['temperature'] != 0.1:
        errors.append("Temperature should be updated")
    
    # Verify existing fields preserved
    if merged['model_config'].get('existing_field') != 'preserve_me':
        errors.append("Existing field should be preserved")
    if merged['model_config']['nested_config']['deep_setting'] != 'keep_this':
        errors.append("Nested config should be preserved")
    
    # Verify complex fields preserved
    if 'prompts' not in merged:
        errors.append("Prompts should be preserved")
    elif len(merged['prompts']) != 2:
        errors.append("All prompts should be preserved")
    
    if merged['extraction']['complex_nested']['deep_setting'] != 'important_value':
        errors.append("Deep nested settings should be preserved")
    
    # Verify neo4j merge
    if merged['neo4j']['host'] != 'new_host':
        errors.append("Neo4j host should be updated")
    if merged['neo4j'].get('complex_neo4j_setting') != 'preserve_this':
        errors.append("Complex neo4j settings should be preserved")
    
    # Verify new field added
    if 'anti_silo' not in merged:
        errors.append("New anti_silo field should be added")
    
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Deep merge test PASSED - All critical fields preserved, updates applied correctly")
    
    # Show some detailed results
    print("\nDetailed Results:")
    print(f"  - model_config.model: {existing_settings['model_config']['model']} -> {merged['model_config']['model']}")
    print(f"  - model_config.existing_field preserved: {merged['model_config'].get('existing_field')}")
    print(f"  - neo4j.host: {existing_settings['neo4j']['host']} -> {merged['neo4j']['host']}")
    print(f"  - neo4j.complex_neo4j_setting preserved: {merged['neo4j'].get('complex_neo4j_setting')}")
    print(f"  - Prompts preserved: {len(merged['prompts'])} prompts")
    print(f"  - Extraction settings preserved: {len(merged['extraction'])} extraction settings")
    print(f"  - Anti-silo added: {'anti_silo' in merged}")
    
    return True

if __name__ == "__main__":
    print("Knowledge Graph Settings Deep Merge Test")
    print("=" * 50)
    
    success = test_deep_merge()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    if success:
        print("✅ Deep merge logic works correctly!")
        print("✅ Complex settings are preserved")
        print("✅ Updates are applied to specified fields")
        print("✅ New fields are added without overwriting existing data")
        print("\n🎉 The fix will prevent knowledge graph data corruption!")
    else:
        print("❌ Deep merge logic has issues that need to be fixed")