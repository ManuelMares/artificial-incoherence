# Description
This is the repo of Artificial Incoherence for the Data Mining class.

# Report
Latex in overleaf:
https://www.overleaf.com/2734718238jcktdpyyxdqd
## Midterm report
    October 22, midterm report
        • Data
            • What kind of data?
            • Dataset information such as number of instances, number of attributes,
            type of attributes
            • How do you process your data?
            • Which attributes you use and which one you don’t use? Why?
        • Data mining task
            • What task? Classification, Clustering, Anomaly detection,...
        • Progress
            • Which algorithms you have tried?
            • How are the preliminary results (e.g., accuracies, running time)?
        • Are there any challenges or difficulties that you are facing?
        • Schedule

## How to work with overleaf
Latex is a language to construct pdf's in a similar way we produce code.
Text files and images can be imported into our main document, allowing for modularity and therefore, easy edition.


### How to reference objects
There are multiple modules that we can use in our Latex documents.
Notice that create an object and add it to the document is different to reference them.
When the symbol ~ appear, it means that you should not put a space between the previous word and the reference. The space will be created automaticall by ~
Here are the most common referenced ones listed:

.tex files				referenced as \input{name_of_file.tex}
.Figure files			referenced as ~\input{name_of_file.tex}
section					referenced as ~\ref{section:name_of_section}
subsection				referenced as ~\ref{section:name_of_subsection}
subsubsection			referenced as ~\ref{section:name_of_subsubsection}
bibliography(citation)	referenced as ~\cite{name-of-reference}

### To create images
Store your images in the folder 'Figures' in the overleaf website.
A copy of those images is kept in this project under the folder 'Figures' as well as a backup

Use and modify this code snippet to insert an image in the text in the exact place where you want it to be

\begin{figure}
    \includegraphics[width=0.55\textwidth]
	{Figures/histograms_bias.png} 					% Route to image
    \caption{
        \textbf{Biased histograms}\\ 				% Title of the image
        This is the histogram of the dataset		% Caption of the image
    }
    \label{figure:bias_histograms}					% Name used to reference the image
\end{figure}

### To create sections
It is enough to write:

\section{name_of_section}				% title to show
\label{section:name_of_section}			%only if this section will be referenced somewhere else

### To add bibliography/citation
A .bib file already exists. Each bibliography item is created as a json object. Just replace the values and indicate a reference name with no space as first value

### To create a list
This is a list with no numbers:
\begin{itemize}  
	\item This is an object
	\item This is a second object
\end{itemize}  

This is a list with numbers:
\begin{itemize}  
	\item[] This is an object
	\item[] This is a second object
\end{itemize}  




# GIT LFS FOR DATA SETS
Due to the size of the involved data sets, it is recommended to work with Git large data sets library. You can install it
by running the following command 'git lfs install' and keeping track of the .gitattributes file.
For further information, check https://git-lfs.com/




# Sources
https://en.www.inegi.org.mx/datosabiertos/
https://www.economia.gob.mx/datamexico/en/vizbuilder


# Schedule
This schedule is just a suggestion
week 1 & 2:
	Data collection. 
We will investigate the different available data sources to have the information as complete as possible

week 3 :
	Data exploration. 
We will analyze the different data sources, we will clean them, normalize them, and will find correlation patterns to create an argument for our research.

week 4,5.
Experimentation.
We will test different approaches, such as different algorithms, different configurations, different visualization methods, etc., to try to determine the most effective methods.

week 6,7. 
	Improvement
Having determined the most effective approach, we will further experiment with its parameter to try to better approximate our model.

week 8,9. 
	Analysis of results
We will analyze our results to try to provide valuable conclusions, as well as produce a systematic report of our approach, limitations and observations.


# Requirements
Python ^2.7. You can either install Python in your computer or set an environment with Anacoda
This project is expected to be run in VSCode

# Working with interactive cells
In order to visualize the the plots, you need to use interactive cells just like the ones in Jupyter Notebook or in Google Collab.
## Create a new cell
add the following comment on top your code to run: '#%%'
## To run the interactive cell
Right above the comment, three buttons will appear. Select 'run', and a new window will appear with your results
## running the whole project in the interactive window
You can running the whole project at once with the rop right buttons, or by right-clicking the document and choosing to run the interactive window. 
The advantage of using cells is not having to process the models all over again every time you do a light modification that does not involve previous steps