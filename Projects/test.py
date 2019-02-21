from Fourier_Basis_Function import Fourier_Basis_Function

f = Fourier_Basis_Function(3,2,5,0.1)

print(len(f.c))
for i in range(100):
    print("Expected:")
    print(f.compute([0.5,0.75,0.5]))
    print("-"*10)
    print("Result After",i+1,"Gradient Steps:")
    f.gradient_step([0.5,0.75,0.5],[2,5])
    print()

