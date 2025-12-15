# Brain2Text25

Intro

## To download data

After downloading the legacy API key from kaggle, run:
```
!kaggle competitions download -c brain-to-text-25
```

and then unzip into a data foler:

linux
```
mkdir data
unzip brain-to-text-25.zip -d data/2
```
windows
```
mkdir data
tar -xf brain-to-text-25.zip -C data/
```

## Ce qu'il se passe (pour l'instant)

brain signals -> CNN-GRU -> phoneme sequences -> Transformer -> english
