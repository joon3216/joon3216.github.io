
# Set a working directory to where this script is located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#render('2019.Rmd')
index <- paste(readLines('2019.html', encoding = "UTF-8"), collapse = '\n')
to_delete <- '<section class=\"page-header\">\n<h1 class=\"title toc-ignore project-name\">Projects: 2019</h1>\n</section>'
index <- gsub(to_delete, '', index)
index <- xml2::read_html(index)
xml2::write_html(index, '2019.html')
