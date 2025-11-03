from spellchecker import SpellChecker

class SpellCheckerApp:
    def __init__(self):
        self.spell = SpellChecker()
    
    def correct_txt(self,txt):
        words = txt.split()
        corrected_words = []
    
        for word in words:
            corrected_word = self.spell.correction(word)
            if corrected_word != word.lower():
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

            
        return ' '.join(corrected_words)
    
    def run(self):
        print("\n ---Spell Checker --")

        while True:
            text = input('Enter text to check or type "exit" to quit: ')

            if text.lower() == 'exit':
                print('Closing the program....')
                break

            corrected_text = self.correct_txt(text)
            print(f'Corrected Text : {corrected_text}')

if __name__ == "__main__":
    SpellCheckerApp().run()
            