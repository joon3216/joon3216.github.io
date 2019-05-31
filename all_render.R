
# Why this script exists: 
# to apply changes in style to all .html files in the repository

library(rmarkdown)
library(purrr)

# Setting a working directory to where this script is located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


c(
    # 00_lvl, or the surface level
    'cv.Rmd',
    'index.Rmd',
    'news.Rmd',
    'projects.Rmd',
    'research_materials.Rmd',



    # 01_lvl
    ## news
    'news/news1.Rmd',

    ## projects
    'projects/2019.Rmd',

    ## research_materials
    'research_materials/2018.Rmd',
    'research_materials/2019.Rmd',

    ## style
    'style/testing.Rmd',



    # 02_lvl
    ## projects
    ### 2019
    'projects/2019/publishing_website.Rmd',

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
    walk(render)






