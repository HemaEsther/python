import random

choices = ['rock', 'paper', 'scissors']
user_score = 0
computer_score = 0
ties = 0

print("🎮 Welcome to Rock, Paper, Scissors!")
print("Type 'quit' anytime to stop.\n")

while True:
    user_choice = input("Enter rock, paper, or scissors (or 'quit' to stop): ").lower()

    if user_choice == 'quit':
        print("\nGame Over! 🏁")
        print(f"Your score: {user_score}")
        print(f"Computer score: {computer_score}")
        print(f"Ties: {ties}")
        print("\nThanks for playing! 👋")
        break

    if user_choice not in choices:
        print("❌ Invalid choice! Please type rock, paper, or scissors.\n")
        continue

    computer_choice = random.choice(choices)
    print(f"\nYou chose: {user_choice}")
    print(f"Computer chose: {computer_choice}\n")

    if user_choice == computer_choice:
        print("It's a tie! 🤝\n")
        ties += 1
    elif user_choice == "rock":
        if computer_choice == "scissors":
            print("You Win! Rock crushes scissors.\n")
            user_score += 1
        else:
            print("You Lose! Paper covers rock.\n")
            computer_score += 1
    elif user_choice == "paper":
        if computer_choice == "rock":
            print("You Win! Paper covers rock.\n")
            user_score += 1
        else:
            print("You Lose! Scissors cut paper.\n")
            computer_score += 1
    elif user_choice == "scissors":
        if computer_choice == "paper":
            print("You Win! Scissors cut paper.\n")
            user_score += 1
        else:
            print("You Lose! Rock crushes scissors.\n")
            computer_score += 1

    print(f"🏆 Scores → You: {user_score} | Computer: {computer_score} | Ties: {ties}\n")
    print("-" * 40)
