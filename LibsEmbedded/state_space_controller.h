
template<unsigned int inputs_count, unsigned int outputs_count, unsigned int controller_order> 
class StateSpaceController
{
    public:
        void init(float *mat_a, float *mat_b, float *smat_c)
        {
            this->mat_a = mat_a;
            this->mat_b = mat_b;
            this->mat_c = mat_c;
        }


        void step(float *output, float *required_input, float *plant_output)
        {
            vmatmul(state_new, state, this->mat_a, 1, controller_order, controller_order, state_new);
            vmatmul(state_new, state, this->mat_b, 1, controller_order, controller_order, state_new);
        }



        void vmatmul(float *result, float *vect, float *mat, unsigned int rows, unsigned int cols)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                float sum = 0.0;
                for (unsigned int k = 0; k < rows; k++)
                {
                    sum+= vect[k]*mat[k*cols + j];
                }

                result[j] = sum;
            }
        }
}