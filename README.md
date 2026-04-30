

For the prints of confusion matrix and erro rates, and the plots that were used in the report, leave everything as it is.
Note that in MNIST, i left GPU=True to make the code run faster. The time that were measured for the report had GPU=False
For tweaking of values:

## Iris.py

In the beginning of main(), you can tweak these values:

    alpha            
    Inverted         
    removed_features 

    num_correct_plots, num_wrong_plots 
    show_K_closest_samples 
    show_difference 



## Iris.py

In the beginning of main(), if you want to tweak some values, first change this to:
    Run_Typical_Examples = False

and then you can tweak these values:
    alpha                    
    Inverted                 
    removed_features  

    plotting_features 
    plotting_MSE