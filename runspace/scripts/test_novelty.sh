uv run python check_novelty.py \
  --training-data ../tokenized_data/TinyStoriesV2-GPT4-train_tokens.npy \
  --tokenizer-path ../tokenizer_models \
  --vocab-filename tokenizer_vocab_tinystories_10k.json \
  --merges-filename tokenizer_merges_tinystories_10k.txt \
  --max-samples 10000 \
  --text "$(cat <<'EOF'
Today is Friday, Mary and her mommy went to the store and Mary saw a big bag of popcorn. She asked her mommy, "Can I have the popcorn, please?" Her mommy smiled and said, "Yes, you can have it."
Mary was so happy! She quickly grabbed the popcorn and started eating it. It was so yummy! She smiled and said, "Yummy! I love it!"
Mary was so happy that she got to eat the popcorn. She was so excited that she ate it all up. She was so full and happy.
The end.
EOF
)"
