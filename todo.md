
## Done
- Extract data
- Basic model GRU: Gated Recurrent Unit (RNN with added gates to avoid short memory but simpler to train than LSTM). Training in about 5 min on cpu
- Model GRU good (2.5), trying to add CNN to extract local temporal patterns (can try Conformer after). Gives 1.9 loss in 15min

## Todo
- Try CNN GRU on more sessions to see if possible to have less than 1.9 loss (accuracy should be around 40%, not that good)
- Add traduction, n-gram phomones to sentence 
- Connect Timeseries model to translator