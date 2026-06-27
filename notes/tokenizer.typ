== Tokenizer

Why do we need tokenizer in NLP anyways?

I like to think NLP tokenizer like lexical analysis in compiler building. Text are just meaning less bytes, a stream of data with no structure at all.

Tokenizer a process to make this data into something manage-able.

A good way to start tokenize, probably try to split the text in to a list of characters.

```
"hello world" -> ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
```

Cool, but that make the token list a bit long, not great for performance.

A another good way would be split by word.
```
"hello world" -> ['hello', 'world']
```

But the number of words is infinite, new words would just be invented or emerge

- For example, "brain rot" is a recent word, should you just add next word and go re-train your model?

- Or user input a typo "morningg", would the model should re-train now?

Fortunately, new words still combine from characters, or better, subword
- cats -> 'cat', 's'
- "brain rot" -> 'brain' 'rot'

So we can use something in the middle, between characters and words: subtokens
- good performance
- solve 'unknown words" problem

