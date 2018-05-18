#!/bin/bash
wget -O 'vae_history.p' 'https://www.dropbox.com/s/az77k0697jrks7t/history.p?dl=1'
wget -O 'vae.h5' 'https://www.dropbox.com/s/d4or3yf7zx1qqi7/vae.h5?dl=1'
wget -O 'vae_generator.h5' 'https://www.dropbox.com/s/xz6fbxxhi0tn5je/generator.h5?dl=1'
wget -O 'vae_encoder.h5' 'https://www.dropbox.com/s/igqq1m98g9w6okd/encoder.h5?dl=1'
wget -O 'gan_history.p' 'https://www.dropbox.com/s/gh65vr50y8h4j4w/gan-history.p?dl=1'
wget -O 'gan_generator.h5' 'https://www.dropbox.com/s/unptdd9ag9d2aw1/params_generator_epoch_000027.h5?dl=1'
wget -O 'acgan_history.p' 'https://www.dropbox.com/s/ydg89e5lk8uhf5h/acgan-history.p?dl=1'
wget -O 'acgan_generator.h5' 'https://www.dropbox.com/s/iah7m5s5abz9li4/plot_epoch_043_generated.png?dl=1'
python3 output.py $1 $2