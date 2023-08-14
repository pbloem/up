# Universal pre-training

## Installation instructions

First install the [former]() package. Download or clone it into a separate dir and install with `pip install -e .` from the root directory. **Dot not copy it to the root directory of the up repository**.

Then, download or clone the `up` repository (again, into a separate directory). Install with `pip install -e .` from the root directory.

## Potential claims
 - UP allows for better than chance zero-shot performance.
   - Make nice plots with baseline bars.
   - TODO: Show for image setting. 
 - UP allows a model to reach better performance in 24 hrs than with traditional initialization. (Not counting the time it took to perform UP) 
   - GTP-style, on wikipedia 10^9
   - BERT Cramming.
 - Even including the time it takes to perform UP, we get better performance (doubtful, but worth a shot)
   - With and without the sampling time (sampling could be amortized over multiple models).
   - Try for different amounts of pre-training: eg 4/20, 8/16, 12/12, 16/8, 20/4. Plot. 
 - **UP leads to better performance with less data.**
   - The other experiments should provide a hint. Try with different subsets.
   - This is central to the claim we want to make that UP can help to solve some of the problems we face today (like copyrighted data, and poorly curated data).
 - UP Leads to better OOD generalization
   - Try some other English, non-wp validation sets: Alice, Brittanica, Dialogue data. Some other languages? 

## Sequential version:
 - add a random sequence of tokens to condition on, using cross-attention.
 - Follow the same proof as with the non-sequential version.
 - 

## To Experiment:
 - Does sampling the temperature improve things, or is a low, tuned temperature better?
 - Should I reset the optimizer and re-warmup, or just use the same Adam?
 - What does the algorithm look like in the limit of single batch iterations? A kind of self distillation with the last
   6 layers re-initialized every step?
 - Try relative embedings.

## General notes and reminders
 - For the computation source, non-linearities that have poort gradient propagation seem to work pretty well (at least in the image domain). So far, I've only tried sign and a sharp sigmoid. Maybe a sharp tanh?
   - In general, it bears keeping in mind that most of our modern-day components are designed first and foremost for good gradient propagation, and that is not necessarily what we're looking for here.  
 - The wikipedia data should be loaded _as bytes_. The file is in unicode, and we represent this by loading it as a byte stream, and loading unicode characters as two bytes.