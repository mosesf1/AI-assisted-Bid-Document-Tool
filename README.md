# AI-assisted-Bid-Document-Tool
Simple python code that utilizes different LLM's to extract and summarize bid documents.
How to use the code:
1. Using your python IDE like pycharm etc. download GPT_extract.py and key.env, put both files in the same python project
2. In the .env file where you see "Insert key here" place your OpenAI API key, I will not provide mine as it is a security risk, and be warned it will cost you tokens each time you use the program. It is approximatley a dollars worth of tokens each use.
3. Press run on the python file, it will prompt you to select the pdf you would like to summarize and extract, select the PDF
4. It will return a PDF, Json, and md file, for simplicity you may view the PDF file for the summary. The runtime was approx 100 seconds in testing, and there is an issue with OpenAI where at times dashes may appear as black boxes so keep that in mind(a better model with anthropic could theoraticely be made, but because of the cost of running the keys we were limnited in testing)
5. Finally you have all the files needed, to reuse for a different PDF just run the program and when prompted select a different contract bid
