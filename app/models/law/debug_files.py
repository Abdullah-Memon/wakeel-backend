import re
import os

def inspect_legal_files():
    """Inspect the actual content of legal files to identify patterns"""
    
    file_paths = {
        "constitution": "app/models/law/data/Constitution Articles.txt",
        "ppc": "app/models/law/data/PPC_sections.txt",
    }
    
    for doc_type, file_path in file_paths.items():
        print(f"\n{'='*60}")
        print(f"INSPECTING {doc_type.upper()} FILE")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            print(f"✅ File loaded: {len(content)} chars, {len(lines)} lines")
            
            # Show first 20 non-empty lines
            print("\n📄 FIRST 20 NON-EMPTY LINES:")
            print("-" * 40)
            line_count = 0
            for i, line in enumerate(lines):
                if line.strip() and line_count < 20:
                    print(f"{i:3d}: {line.strip()}")
                    line_count += 1
            
            # Test various patterns
            print(f"\n🔍 PATTERN ANALYSIS FOR {doc_type.upper()}:")
            print("-" * 40)
            
            if doc_type == "constitution":
                patterns = [
                    (r'^\s*آرٹیکل\s*(\d+)', "آرٹیکل pattern"),
                    (r'^\s*Article\s*(\d+)', "Article pattern"),
                    (r'^\s*(\d+)[:\.]\s*', "Number: pattern"),
                    (r'آرٹیکل', "Contains آرٹیکل"),
                    (r'Article', "Contains Article"),
                ]
            else:
                patterns = [
                    (r'^\s*سیڪشن\s*(\d+)', "سیڪشن pattern"),
                    (r'^\s*Section\s*(\d+)', "Section pattern"),
                    (r'^\s*(\d+)[:\.]\s*', "Number: pattern"),
                    (r'سیڪشن', "Contains سیڪشن"),
                    (r'Section', "Contains Section"),
                ]
            
            for pattern, description in patterns:
                matches = []
                for i, line in enumerate(lines[:100]):  # Check first 100 lines
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append((i, line.strip()))
                
                print(f"  {description}: {len(matches)} matches")
                if matches:
                    for i, (line_num, line_text) in enumerate(matches[:3]):
                        print(f"    {line_num:3d}: {line_text[:80]}...")
            
            # Character encoding check
            print(f"\n📝 CHARACTER ENCODING CHECK:")
            print("-" * 40)
            sample_text = content[:500]
            print(f"Sample text: {sample_text}")
            print(f"Urdu/Arabic chars found: {bool(re.search(r'[\u0600-\u06FF]', sample_text))}")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    inspect_legal_files()