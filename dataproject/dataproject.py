import matplotlib.pyplot as plt
import ipywidgets as widgets

class GenderPlotter:
    def __init__(self, data, gender_label):
        self.data = data
        self.gender_label = gender_label

    def plot_gender(self, df, ethnicity, age_group):
        I = (df['Ethnicity'] == ethnicity) & (df['Age_group'] == age_group)
        ax = plt.gca()  # Get or create the current Axes instance
        df.loc[I, :].plot(x='Year', y='Income', style='-o', legend=False, ax=ax)
        
        # Add description to the plot
        plt.text(0.5, 0.95, self.gender_label, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        
        # Add axis labels
        plt.xlabel('Year')
        plt.ylabel('Mean income before taxes, kr.')
        
        plt.show()

    def interactive_plot(self):
        # Create dropdown widgets for 'Ethnicity' and 'Age_group'
        ethnicity_dropdown = widgets.Dropdown(description='Ethnicity', 
                                              options=self.data['Ethnicity'].unique(), 
                                              value=self.data['Ethnicity'].iloc[0])
        age_group_dropdown = widgets.Dropdown(description='Age Group', 
                                              options=self.data['Age_group'].unique(), 
                                              value=self.data['Age_group'].iloc[0])
        # Interactively call the plot function with the selected options
        widgets.interact(self.plot_gender, 
                         df=widgets.fixed(self.data),
                         ethnicity=ethnicity_dropdown,
                         age_group=age_group_dropdown)