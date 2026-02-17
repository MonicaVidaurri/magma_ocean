# DEPENDENCIES
TidalPy
scipy
numpy
tqdm
CyRK

!!!!!!!!!!!!!!!! FIRST, A SUPER DUPER IMPORTANT NOTE !!!!!!!!!!!!!!!
Right now if you want to change the XUV model you have to manually do so in
utils/get_loss.py. I know, I know - sorry about that.

--------------------------------------------------------------------------
Howdy!!

This model is a python translation of GOOEY, a coupled magma ocean-atmosphere model
designed to track the evolution of oxygen and water in a steam atmosphere undergoing
hydrodynamic escape, developed by Laura Schaefer.

---All you gotta do is change the user input stuff at the top of the run_model.py file,
and if you wanna change between the high/low XUV model, just change that value in
utils/get_loss.py, like it says at the top. The model currently contains stellar data
for our Sun, TRAPPIST-1, and Proxima Centauri. Instructions for making your own stellar
data file are in the text file HOW_TO-make_stellardata.txt. ***The ONLY file you need
to run is run_model.py!!***---

The paper for GOOEY can be found here: http://doi.org/10.3847/0004-637X/829/2/63
and the model can be accessed here: https://purl.stanford.edu/rk050tc3031
For additional reaading + an intercomparison between magma ocean models, check out
what the CHILI team has been cookin: https://arxiv.org/pdf/2511.16142

Feel free to email me at monica.r.vidaurri@nasa.gov (or if that don't work,
monavidaurri@gmail.com) if you've got any questions! I'd highly encourage you to read
Laura's paper first - all the core physics is the same, I just put everything
in python terms. (In other words, I'm not the model expert here lol I'm just
the translator.)

This model doesn't have a name but I don't like the name GOOEY (you can tell
Laura I said that - she already knows). I haven't come up with anything clever
yet, but maybe you can help :) but I ain't callin it GOOEY that's for
damn sure; I just think we can do better.

Good luck and have fun!