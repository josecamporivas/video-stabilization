#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <emmintrin.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#define TAMBLOQUE 32

int esIgual(int filaBloque1, int colBloque1, IplImage * imagen1, int filaBloque2, int colBloque2, IplImage * imagen2){
    
    //recorro los bloques de las respectivas imagenes
    for(int fila = 0; fila < TAMBLOQUE; fila++) {
        __m128i *a = (__m128i *) (imagen1->imageData + ((filaBloque1 + fila) * imagen1->widthStep) + (colBloque1 * imagen1->nChannels));
        __m128i *b = (__m128i *) (imagen2->imageData + ((filaBloque2 + fila) * imagen2->widthStep) + (colBloque2 * imagen2->nChannels));        
        for(int col = 0; col < TAMBLOQUE*imagen1->nChannels; col+=16) {
            //calculo la diferencia entre los punteros
            __m128i A = _mm_loadu_si128(a);                                                                                     
            __m128i B = _mm_loadu_si128(b);                                                                                     
            __m128i aux = _mm_sad_epu8(A, B);                                                                                               
            __m128i aux2 = _mm_srli_si128(aux, 8);
            
            //si hay diferencia entre los punteros devuelvo 0 (false) (punto diferenciador para aumentar la eficiencia del codigo)
            if(_mm_cvtsi128_si32(_mm_add_epi32(aux, aux2)) > 0){
                return 0;
            }  
            //sino continuo
            a++;                                                                                                                         
            b++;                                                                                                                     
        }
    }
    //si recorro los 2 bloques sin ver diferencias, devuelvo 1 (true)
    return 1;
}

void buscarBloque(IplImage *original, IplImage *parecido, int filaMedio, int colMedio, int *numFilaCoinc, int *numColCoinc){
    
    int encontrado = 0;
    
    //empiezo en filaMedio-35
    int rangoFila = -35;
    while(!encontrado && rangoFila < 80){
        //empiezo en colMedio-35
        int rangoCol = -35;
        while(!encontrado && rangoCol < 54){
            
            //si comparar bloque devuelve 1, significa que los bloques son iguales
            if(esIgual(filaMedio, colMedio, original, filaMedio + rangoFila, colMedio + rangoCol, parecido)){
                //como encontrado==1 salgo de los bucles
                encontrado = 1;
                //guardo el desplazamiento
                *numFilaCoinc = rangoFila;
                *numColCoinc = rangoCol;
            }
            rangoCol++;
        }
        rangoFila++;
    }
    
}

__m128i getComponenteColor(uchar *pImagenOriginal, int fila, int col, int numFilaCoinc, int numColCoinc, int filaLimite, int colLimite){
    //result es el valor que tendrá los bordes del frame desplazado (podría cambiarse a otro color facilmente)
    __m128i result = _mm_set1_epi8(0x00);

    //si hay desplazamiento horizontal a la izquierda y estoy en una posicion que afecta, devuelvo result
    if(numColCoinc<0 && col < -3*numColCoinc){  
        return result;
        //si hay desplazamiento vertical hacia arriba y estoy en una posicion que afecta, devuelvo result
    }else if(numFilaCoinc<0 && fila < -1*numFilaCoinc){
        return result;
        //si hay desplazamiento vertical hacia abajo y estoy en una posicion que afecta, devuelvo result
    }else if(filaLimite - numFilaCoinc < fila){
        return result;
        //si hay desplazamiento horizontal a la derecha y estoy en una posicion que afecta, devuelvo result
    }else if(colLimite - 4*numColCoinc < col){
        return result;
    }else{
        //sino devuelvo el valor del puntero aplicando el desplazamiento
        return _mm_loadu_si128((__m128i*)(pImagenOriginal + numColCoinc*3));
    }
}

IplImage* desplazarFrame(IplImage* InFrameNew, int numFilaCoinc, int numColCoinc){
    //creo una copia del frame actual donde guardaré el frame desplazado
    IplImage *frameCopiado = cvCloneImage(InFrameNew);
    //recorro los frames
    for(int fila = 0; fila < InFrameNew->height; fila++){
        //la copia la recorro de forma normal
        __m128i *pImagenCopia = (__m128i *)(frameCopiado->imageData + fila * frameCopiado->widthStep);
        //el frame sin desplazar lo recorro situandome el la fila correspondiente al desplazamiento que tiene
        __m128i *pImagenOriginal = (__m128i *) (InFrameNew->imageData + (fila + (numFilaCoinc<0 && fila < -1*numFilaCoinc? 0: numFilaCoinc))*InFrameNew->widthStep);
        for(int col=0; col < InFrameNew->widthStep; col+=16){
            //calculo el valor que deberia tener el puntero del frame que voy a desplazar(o color negro o valor de *pImagenOriginal)
            _mm_storeu_si128(pImagenCopia, getComponenteColor((uchar *)pImagenOriginal, fila, col, numFilaCoinc, numColCoinc, frameCopiado->height, 3*frameCopiado->width));
            pImagenCopia++;
            pImagenOriginal++;
        }
    }    
    //devuelvo el frame desplazado
    return frameCopiado;
}

int main(int argc, char** argv) { 
    if (argc < 2 || argc > 3)
        exit(-1);
    
    //codigo para leer el video a partir del segundo parametro o del tercero, dependiendo si hay comando -showoff
    //interpreto que la direccion del video siempre va a estar en el último parametro
    int mostrarVentana = 1;
    int posDirVideo = 1;
    
    if(!strcmp(argv[1], "-showoff")){
        mostrarVentana = 0;
        posDirVideo = 2;
    }
    // Creamos las imagenes a mostrar
    CvCapture* capture = cvCaptureFromAVI(argv[posDirVideo]);

    // Always check if the program can find a file
    if (!capture) {
        printf("Error: fichero %s no leido\n",argv[posDirVideo]);
        return EXIT_FAILURE;
    }

    IplImage *InFrameNew;
    //guardo el primer frame para poder compararlo con el resto y obtener el desplazamiento del frame
    IplImage *firstFrame = cvCloneImage(cvQueryFrame(capture));
    
    //localizacion del bloque que voy a comparar
    int filaMedio = (firstFrame->height)/2;
    int colMedio = (firstFrame->width)/2;

    //varialbes donde guardo el desplazamiento del frame
    int numFilaCoinc = 99;
    int numColCoinc = 99;
    
    //codigo para medir el tiempo de ejecucion del programa
    struct timespec start, finish;
    float elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while ((InFrameNew = cvQueryFrame(capture)) != NULL) {
        //obtengo las coordenadas del bloque buscado
        buscarBloque(firstFrame, InFrameNew, filaMedio, colMedio, &numFilaCoinc, &numColCoinc);
        
        //guardo el frame con el desplazamiento aplicado en el frame actual
        InFrameNew = desplazarFrame(InFrameNew, numFilaCoinc, numColCoinc);
        
        //si el comando -showoff no está, muestro la ventana
        if(mostrarVentana){
            cvShowImage("Frame Video", InFrameNew);
            cvWaitKey(1);
        }
    }
    //muestro el tiempo de ejecucion
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
//    elapsed = (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Tiempo transcurrido: %f segundos", elapsed);
}