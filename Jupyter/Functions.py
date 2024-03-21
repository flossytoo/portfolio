import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import matplotlib.pyplot as plt
import seaborn as sns


def backward_selected_log_reg(data, response):
    """Logistic model designed by backward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels logistic model
           with an intercept
           selected by backward selection
           evaluated by parameters p-value
    """
    remaining = set(data._get_numeric_data().columns)
    if response in remaining:
        remaining.remove(response)
    cond = True

    while remaining and cond:
        formula = "{} ~ {} + 1".format(response, ' + '.join(remaining))
        print('_______________________________')
        print(formula)
        model = smf.logit(formula, data).fit()
        score = model.pvalues[1:]
        toRemove = score[score == score.max()]
        if toRemove.values.max() > 0.05:
            print('remove', toRemove.index[0], '(p-value :', round(toRemove.values[0], 3), ')')
            remaining.remove(toRemove.index[0])
        else:
            cond = False
            print('is the final model !')
        print('')

    # Move outside the loop to print summary only for the final model
    print(model.summary())
    
    return model

def backward_selected(data, response):
    """Linear model designed by backward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by backward selection
           evaluated by parameters p-value
    """
    remaining = set(data._get_numeric_data().columns)
    if response in remaining:
        remaining.remove(response)
    cond = True

    while remaining and cond:
        formula = "{} ~ {} + 1".format(response,' + '.join(remaining))
        print('_______________________________')
        print(formula)
        model = smf.ols(formula, data).fit()
        score = model.pvalues[1:]
        toRemove = score[score == score.max()]
        if toRemove.values > 0.05:
            print('remove', toRemove.index[0], '(p-value :', round(toRemove.values[0],3), ')')
            remaining.remove(toRemove.index[0])
        else:
            cond = False
            print('is the final model !')
        print('')
    print(model.summary())
    
    return model

def graph_box_plots(df, columns):
    labels = ['Faux', 'Vrai']
    num_plots = len(columns)
    num_rows = math.ceil(num_plots / 3)
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        ax = axes[i]
        sns.boxplot(data=df, x="is_genuine", y=column, ax=ax)
        ax.set_title(f'Box Plot for {column} for real and fake banknotes')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)

    plt.tight_layout()
    plt.show()     