import text
import image
import os
import sys

def print_banner():
    print("="*40)
    print("   AI FORENSIC TERMINAL v2.0 (2026)   ")
    print("="*40)

def main():
    print_banner()
    print("1. Analyze Text Content")
    print("2. Analyze Image Authenticity")
    print("3. Exit")
    
    choice = input("\nSelect Option: ")

    if choice == '1':
        user_input = input("\nEnter text to analyze:\n> ")
        if len(user_input.strip()) < 10:
            print("Error: Text too short for reliable analysis.")
            return
            
        res = text.predict_text(user_input)
        b = text.get_burstiness(user_input)
        
        print("\n[TEXT ANALYSIS RESULTS]")
        print(f"Human Probability: {res['Human']:.1%}")
        print(f"AI Probability:    {res['AI']:.1%}")
        print(f"Post-Edited AI:   {res['Edited']:.1%}")
        print(f"Burstiness Score:  {b}")

    elif choice == '2':
        path = input("\nDrag and drop image or enter path: ").strip().replace('"', '')
        if os.path.exists(path):
            print("Scanning pixels for AI artifacts...")
            try:
                ai_chance = image.predict_image(path)
                print(f"\n[IMAGE ANALYSIS RESULTS]")
                print(f"AI Generation Chance: {ai_chance:.2f}%")
                if ai_chance > 50:
                    print("Status: ⚠️ PREDICTED AI GENERATED")
                else:
                    print("Status: ✅ PREDICTED REAL/HUMAN")
            except Exception as e:
                print(f"Error: Could not process image. {e}")
        else:
            print("Error: File path does not exist.")

    elif choice == '3':
        sys.exit()

if __name__ == "__main__":
    while True:
        main()
        if input("\nRun another scan? (y/n): ").lower() != 'y':
            break