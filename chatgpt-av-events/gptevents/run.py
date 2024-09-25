# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import gptevents as gpte

gpte.logs(show_level='info', show_color=True)
logger = gpte.CustomLogger(__name__)  # use custom logger

# const
SAVE_P = True  # save pickle files with data
LOAD_P = False  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = True  # filter GPT4-V and heroku data
CLEAN_DATA = True  # clean GPT4-V data
ANALYSE_DATA = True  # analyse GPT4-V data
SHOW_OUTPUT = True  # should figures be plotted


if __name__ == '__main__':
    # create object for working with heroku data
    reports = gpte.common.get_configs('reports')
    chatgpt = gpte.analysis.ChatGPT(files_reports=reports, save_p=SAVE_P, load_p=LOAD_P, save_csv=SAVE_CSV)
    # read heroku data
    data = chatgpt.read_data(filter_data=FILTER_DATA, clean_data=CLEAN_DATA, analyse_data=ANALYSE_DATA)
    logger.info('Data from {} reports included in analysis.', data.shape[0])
    if SHOW_OUTPUT:
        # Output
        analysis = gpte.analysis.Analysis()
        logger.info('Creating figures.')
        # some bar plot
        analysis.bar(data, y=['report'], pretty_text=True, save_file=True)
        # some scatter plot
        analysis.scatter(data, x='report', y='response', color='report', pretty_text=True, save_file=True)
        # # some histogram
        analysis.hist(data, x=['report'],  pretty_text=True, save_file=True)
        # # some map
        # analysis.map(data, color='', save_file=True)
        # check if any figures are to be rendered
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
