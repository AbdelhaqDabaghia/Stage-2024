directory_name_gen = "C:/Users/adabag01/Documents/Stage2024/antoine_a_traiter/12_19_plan_cone_inf";

subdirectory_name = "12_19_plan_cone_inf/alpha_5/1/";
directory_name = directory_name_gen+subdirectory_name;
nb_files=120;

img10_prefix="image";


for(i=1; i<=nb_files; i++)
{
if (i<=9) img_name = img10_prefix+d2s(i, 0);
else
if (i<100) img_name = img100_prefix + d2s(i, 0);

open(directory_name+img_name+".png");
setOption("BlackBackground", true);
run("Convert to Mask");
run("Invert");

saveAs("Tiff", directory_name+img_name+".tif");
}


