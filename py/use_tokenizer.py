import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import loggy3
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from loggy3 import chunk_tokenizer, token_to_readable, print_token_sequence
    
    def main():
        # Example usage
        if len(sys.argv) < 3:
            print("Usage: python use_tokenizer.py <chunk_path> <os_type>")
            print("Example: python use_tokenizer.py ./data/chunk_1 Mac")
            sys.exit(1)
            
        chunk_path = sys.argv[1]
        os_type = sys.argv[2]  # "Mac" or "Windows"
        
        if not os.path.exists(chunk_path):
            print(f"Error: Path {chunk_path} does not exist")
            sys.exit(1)
            
        print(f"Processing chunk at {chunk_path} for OS {os_type}")
        
        try:
            tokens = chunk_tokenizer(chunk_path, os_type)
            print(f"Found {len(tokens)} tokens")
            
            # Option 1: Use print_token_sequence to print all tokens
            print("\nPrinting all tokens:")
            print_token_sequence(tokens)
            
            # Option 2: Display first 10 tokens with human-readable descriptions
            print("\nFirst 10 tokens with descriptions:")
            for i, (seq, token) in enumerate(tokens[:10]):
                readable = token_to_readable(token)
                print(f"Token {i+1}: Sequence {seq}, Value {token} - {readable}")
            
            # Option 3: Filter and print only keyboard events
            print("\nDemo: Filtering for keyboard events only:")
            keyboard_events = [(seq, token) for seq, token in tokens if "🔑" in token_to_readable(token)]
            if keyboard_events:
                print(f"Found {len(keyboard_events)} keyboard events")
                for seq, token in keyboard_events[:5]:
                    print(f"[{seq}] {token_to_readable(token)}")
            else:
                print("No keyboard events found")

            print(type(tokens), type(tokens[0]), type(tokens[0][0]), type(tokens[0][1]))
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing loggy3 module: {e}")
    print("Make sure you've built the Rust extension with `maturin develop` or `python setup.py develop`") 