# main.py

import os


# =====================================================
# MENU
# =====================================================

def menu():

    print("\n========================================")
    print("🚀 HUMAN ACTIVITY RECOGNITION SYSTEM")
    print("========================================\n")

    print("1. Train Baseline Models")
    print("2. Train Tuned Models")
    print("3. Evaluate Best Model")
    print("4. Compare All Models")
    print("5. Predict Activity")
    print("6. Launch GUI Dashboard")
    print("7. Exit")


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    menu()

    choice = input("\nEnter your choice: ")


    # -------------------------------------------------
    # BASELINE TRAINING
    # -------------------------------------------------
    if choice == "1":

        print("\n🔵 Training baseline models...\n")
        os.system("python -m src.train_baseline")


    # -------------------------------------------------
    # TUNED TRAINING
    # -------------------------------------------------
    elif choice == "2":

        print("\n🧠 Training tuned models...\n")
        os.system("python -m src.train_tuned")


    # -------------------------------------------------
    # EVALUATION (BEST MODEL)
    # -------------------------------------------------
    elif choice == "3":

        print("\n📊 Evaluating best model...\n")
        os.system("python -m src.evaluate")


    # -------------------------------------------------
    # COMPARE ALL MODELS
    # -------------------------------------------------
    elif choice == "4":

        print("\n📊 Comparing all models...\n")
        os.system("python -m src.evaluate_all")


    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    elif choice == "5":

        print("\n🔮 Running prediction...\n")
        os.system("python -m src.predict")


    # -------------------------------------------------
    # STREAMLIT GUI
    # -------------------------------------------------
    elif choice == "6":

        print("\n🖥️ Launching GUI dashboard...\n")
        os.system("python -m streamlit run gui/app.py")

    # -------------------------------------------------
    # EXIT
    # -------------------------------------------------
    elif choice == "7":

        print("\n👋 Exiting project...")
        break


    # -------------------------------------------------
    # INVALID INPUT
    # -------------------------------------------------
    else:

        print("\n❌ Invalid choice. Try again.")