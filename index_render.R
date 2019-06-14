
# Set a working directory to where this script is located
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# rmarkdown::render('index.Rmd')
index <- paste(readLines('index.html', encoding = "UTF-8"), collapse = '\n')
to_delete <- '<section class=\"page-header\">\n<h1 class=\"title toc-ignore project-name\">Junkyu Park</h1>\n</section>'
index <- gsub(to_delete, '', index)
index <- xml2::read_html(index)
xml2::write_html(index, 'index.html')
