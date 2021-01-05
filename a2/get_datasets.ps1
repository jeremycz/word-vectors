$DATASETS_DIR="utils/datasets"
New-Item -ItemType Directory -Force -Path $DATASETS_DIR

cd $DATASETS_DIR

# Get Stanford Sentiment Treebank
Invoke-WebRequest http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip -OutFile stanfordSentimentTreebank.zip
Expand-Archive -Path stanfordSentimentTreebank.zip -DestinationPath .
Remove-Item -Path stanfordSentimentTreebank.zip

cd ..\..