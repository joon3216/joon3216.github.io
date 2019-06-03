
# Set a working directory to where this script is located
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#render('projects.Rmd')
index <- paste(readLines('projects.html', encoding = "UTF-8"), collapse = '\n')
to_delete <- '<section class=\"page-header\">\n<h1 class=\"title toc-ignore project-name\">Projects</h1>\n</section>'
index <- gsub(to_delete, '', index)
index <- xml2::read_html(index)
xml2::write_html(index, 'projects.html')
