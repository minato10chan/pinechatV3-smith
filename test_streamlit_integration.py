#!/usr/bin/env python3
"""
Integration test for Streamlit app with environment variables
"""
import os
import sys
import subprocess
import time

def setup_test_environment():
    """Set up test environment variables"""
    os.environ['PINECONE_API_KEY'] = 'test-key-for-chunking-only'
    os.environ['OPENAI_API_KEY'] = 'test-key-for-chunking-only'
    os.environ['PINECONE_INDEX_NAME'] = 'test-index'
    os.environ['PINECONE_ASSISTANT_NAME'] = 'test-assistant'

def test_streamlit_import():
    """Test if Streamlit app can be imported without errors"""
    print("=== Testing Streamlit App Import ===")
    
    setup_test_environment()
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        
        from src.components.property_upload import split_property_data
        from src.config.settings import PROPERTY_MAX_TOKENS
        
        print(f"✅ Successfully imported property_upload module")
        print(f"✅ PROPERTY_MAX_TOKENS = {PROPERTY_MAX_TOKENS}")
        
        test_property = {
            'property_name': 'テスト物件',
            'property_type': 'マンション',
            'prefecture': '東京都',
            'city': '渋谷区',
            'detailed_address': '渋谷1-1-1',
            'property_details': 'これは短いテスト用の物件説明です。' * 100,
            'latitude': '35.6580',
            'longitude': '139.7016'
        }
        
        chunks = split_property_data(test_property)
        print(f"✅ Chunking test successful: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streamlit_import()
    if success:
        print("🎉 Streamlit integration test passed!")
    else:
        print("❌ Streamlit integration test failed!")
    sys.exit(0 if success else 1)
