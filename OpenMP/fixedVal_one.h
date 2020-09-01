class fixedVal{
	public:   //Access specifier
	short int scale = 1;
    short int T= 150;
    signed short int t_start = -20;
    short int t_end = 20;
    
    short int pixel_size = 28; //value can be varied depending on image dimension
    short int m = pixel_size * pixel_size; //Total number of input neurons (layer one)
    
    short int n = 3; //number of neurons in layer 2
    
    signed short int Pmin = -5.0 * scale;
    short int Pth = scale * 50;
    short int Pref = 0;
    short int Prest = 0;
    float D = 0.75 * scale;
    
    float w_max = 2.0 * scale;
    float w_min = -1.2 * scale;
    float sigma = 0.02;
    float A_minus = 0.3;  //when time diffence is negative 
    float A_plus = 0.8;   //when time difference is positive
    short int tau_minus = 10;
    short int tau_plus = 10;
    short int epoch = 20;
    
    short int fr_bits = 12;


};
