
# Why this script exists: 
# to apply changes in style to all .html files in the repository

library(itertools2)
library(magrittr)
library(rmarkdown)


# Set a working directory to where this script is located

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Knit the entire .Rmd files in the repository

c(
    # 00_lvl, or the surface level
    'index.Rmd',
    'news.Rmd',
    'projects.Rmd',
    'research_materials.Rmd',



    # 01_lvl
    ## news
    'news/2019-06.Rmd',
    'news/2019-07.Rmd',
    'news/2019-08.Rmd',

    ## projects
    'projects/2019.Rmd',

    ## research_materials
    'research_materials/2018.Rmd',
    'research_materials/2019.Rmd',

    ## style
    'style/testing.Rmd',
    'style/testing_architect.Rmd',
    'style/testing_hpstr.Rmd',
    'style/testing_tactile.Rmd',



    # 02_lvl
    ## projects
    ### 2019
    'projects/2019/funpark.Rmd',
    'projects/2019/publishing_website.Rmd',
    'projects/2019/sudsoln.Rmd',

    ## research_materials
    ### 2018
    'research_materials/2018/censored_data.Rmd',
    'research_materials/2018/non_separable_penalty.Rmd',
    'research_materials/2018/pgf.Rmd',
    'research_materials/2018/pgf_python.Rmd',
    'research_materials/2018/visualizing_confidence_regions.Rmd',

    ### 2019
    'research_materials/2019/cross_validation_fs.Rmd',
    'research_materials/2019/em_imputation.Rmd',
    'research_materials/2019/em_imputation_python.Rmd',
    'research_materials/2019/frog.Rmd',
    'research_materials/2019/largest_roll.Rmd',
    'research_materials/2019/matrix_derivatives.Rmd',
    'research_materials/2019/multivariate_response.Rmd'
) %>%
    imap(function(x){render(x)}, .) %>%
    as.list()


# Edit some .html files after rendering
# These scripts are in the same directory as all_render.R
c(
    'index_render.R',
    'projects_render.R'
) %>%
    imap(function(x){source(x)}, .) %>%
    as.list()


# These scripts are not in the same directory as all_render.R
phrases_to_delete <- c(
    '<section class=\"page-header\">\n<h1 class=\"title toc-ignore project-name\">Projects: 2019</h1>\n</section>'
)

c(
    'projects/2019.html'
) %>% imap(
    function(html_dir, to_delete) {
        orig <- paste(readLines(html_dir, encoding = "UTF-8"), collapse = '\n')
        orig <- gsub(to_delete, '', orig)
        orig <- xml2::read_html(orig)
        xml2::write_html(orig, html_dir)
    },
    ., phrases_to_delete
) %>% as.list()





