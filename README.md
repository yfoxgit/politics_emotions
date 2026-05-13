[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/iZmZcr4l)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23135464)

# How emotive are politicians?
This project analyses the emotiveness of UK politicians relative to civil servants, by applying the pre-trained model from Gennaro & Ash (2022) to UK government department annual reports and accounts forewords.

(NB: This work was originally submitted as part of an assessment for the MPhil module D200.)

## Set-Up Instructions
To set up the repo, run the following commands.

Note these commands are for a Linux setup, assuming you have ```uv``` installed. Adjustments may be required for MacOS or Windows setups or different package managers.

1. Navigate to desired working directory
```shell
cd /path/to/wkd
```
2. Clone the repository from GitHub
```shell
git clone link-copied-from-GitHub
```
3. Navigate to the repository
```shell
cd /path/to/repo
```
4. Create the environment and install dependencies
```shell
uv sync
```
5. Activate the environment (emotiveness)
```shell
source .venv/bin/activate
```
6. Run the project
```shell
python main.py
```
7. Deactivate the environment
```shell
deactivate
```

## Project slides and report
The project slides and report are saved in the repository under:
* Slides: politicians_emotions_slides.pptx
* Report: politicians_emotions_report.pdf

## Cloning the repo and Git LFS
Due to the large sizes of the pre-trained model files, Git Large File Storage was used to upload the repository contents onto GitHub. Therefore, when replicating the code, the repository must be cloned to run. Downloading the repository as a .zip file will not download the PDFs or model files in a usable manner due to the way Git LFS handles these files.

For more information on Git LFS, visit: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

## Data and Results
The raw PDFs are saved under ```/data/raw```. A full list of sources can be found in the References of the project report.

Processed data and results are already saved in  ```/data``` and ```/results```.

When running the code, if you wish to save data or results, lines 36, 61 and/or 152-157 should be uncommented by deleting ```#``` or ```'''``` in ```main.py```.

## Exploratory Data Analysis
Exploratory Data Analysis was conducted in the ```eda.ipynb``` notebook.

If you wish to save the plot, you can set ```save_plot=True``` in the ```plot_years_authors``` function, and may set a filepath (otherwise it will save to the current directory).

## Claude sentiment analysis
The Claude scores generated using ```sentiment.py``` cannot be replicated without a credited ```ANTHROPIC_API_KEY```.
Because of this, pre-saved results are saved under ```/results/doc_ratings.parquet``` and read in in line 171 in ```main.py``` before further analysis is conducted.

If you wish to run the Claude sentiment analysis yourself, you must create a ```.env``` file containing ```ANTHROPIC_API_KEY="your-Anthropic-key"``` and save this in the repository directory.
You can then uncomment lines 18 and 164-168 and comment out lines 171-172 in ```main.py``` before running the code.

Note that your results may differ due to the innate stochasticity of Large Language Models.
