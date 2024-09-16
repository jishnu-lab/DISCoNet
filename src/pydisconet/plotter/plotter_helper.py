import spacy, matplotlib, re
class _HexCodeFromFrequencyDict():
    '''
    A custom class to pass to the wordcloud package as a color map.
    The package by default colors words randomly.
    This ensures the words are colored based on a given weight value.
    '''
    def __init__(self,frequency_dict,cmap="viridis"):
        self.frequency_dict = frequency_dict
        self.cmap = matplotlib.colormaps.get_cmap(cmap)
        self.max_val = max([value for key,value in self.frequency_dict.items()])

    def __call__(self,word,font_size,position,orientation,random_state=None,**kwargs):
        r, g, b, alpha = self.cmap(self.frequency_dict[word] / self.max_val) # Colours are converted to HEX codes to ensure they are encoded correctly when saved as an svg file
        return matplotlib.colors.rgb2hex(self.cmap(self.frequency_dict[word] / self.max_val), keep_alpha=False)