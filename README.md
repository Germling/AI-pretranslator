Annoyed that your CAT tool claims to offer "AI translation" but doesn't support system prompts?
This script offers a simple workaround, so you can write your own prompt in the sysprompt.txt file.
Then download your project from your CAT tool as TMX or XLIFF and run the script on it.

I suggest starting with a small sample of your translation project to refine your system prompt and, once you're satisfied with the sample results, translating the whole text.

**Advantages**
Simple customization: Custom translation style and terminology injection via system prompt

**Note that you need an API key for OpenAI (comes with paid subscription) and Python (e.g. via www.anaconda.org).**

Usage: 
1. After installing Anaconda, in your Anaconda prompt type "pip install openai" to enable your OpenAI API in Python.
2. Get your OpenAI API key from https://platform.openai.com/api-keys
3. To make your key available in your script, set it as global variable by typing: setx OPENAI_API_KEY "your_api_key_here"
4. Put all files in the same folder on your computer.
5. Customize the sysprompt.txt to your needs. I've added some rules as an example.
6. Run the script on your TMX or XLIFF file by typing: python translate_tmx_xliff.py --input your_tmx_file.tmx --sysprompt-file sysprompt.txt --model gpt-5 --chunk-size 20000
7. The output file is stored in the same folder and named "input_translated" by default. Upload the output file back into your CAT tool and post-edit the results there.

You can also choose other models like gpt-4o (refer to OpenAI documentation for details).
You can set the chunk size accordingly. The chunk size determines how many tokens (input and output) can go through each API call.
Since the script runs segment by segment, you don't need a huge chunk size, but I set it high because I am using a long system prompt with a CSV glossary inside.
Since most glossaries aren't huge, I opted to add it inside the prompt.
