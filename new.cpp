//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#include <iostream>

#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

#if defined(__APPLE__)
// vertex shader in GLSL
const char *vertexSource = R"(
	#version 140
    precision highp float;

	in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	out vec2 texcoord;			// output attribute: texture coordinate

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 140
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";
#else
// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

	in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	out vec2 texcoord;			// output attribute: texture coordinate

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";
#endif

#define EPSILON 0.01


struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

    vec4 operator+(vec4 other) {
        return vec4(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2], v[3] + other.v[3]);
    }

    vec4 operator-(vec4 other) {
        return vec4(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2], v[3] - other.v[3]);
    }

    vec4 operator*(vec4 other) {
        return vec4(v[0] * other.v[0], v[1] * other.v[1], v[2] * other.v[2], v[3] * other.v[3]);
    }

    vec4 operator/(vec4 other) {
        return vec4(v[0] / other.v[0], v[1] / other.v[1], v[2] / other.v[2], v[3] / other.v[3]);
    }

    vec4 operator*(float num) {
        return vec4(v[0] * num, v[1] * num, v[2] * num, v[3] * num);
    }
};

class Vector3 {
public:
	float x, y, z, w;

    Vector3() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }

	Vector3(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}

	bool operator==(Vector3 other) {
		if (this->x != other.x) {
			return false;
		}
		if (this->y != other.y) {
			return false;
		}
		if (this->z != other.z) {
			return false;
		}
		return true;
	}

    Vector3 operator+(const Vector3 v) {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    Vector3 operator-(const Vector3 v) {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }

    Vector3 operator*(float f) {
        return Vector3(x * f, y * f, z * f);
    }

    float operator*(Vector3 v) {
        return x * v.x +  y * v.y + z * v.z;
    }
   
   Vector3 operator%(const Vector3 v) {     // cross product
    return Vector3(y*v.z-z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
   }

    float length() { return (float) sqrt(x * x + y * y + z * z); }

    Vector3 normalize(){ return Vector3(this->x, this->y, this->z) * (1 / this->length()); }

    Vector3 getAPerpendicularVector() {
        return Vector3(this->y, (this->x * (-1.0f)), this->z);
    }

    float t() {
        return atanf(y/x);
    }

    float p() {
        return (float) acos(z / length());
    }

};

vec4 SKY_COLOR = vec4(0.8, 0.9, 1, 1);
vec4 SKY_BACKGROUND_COLOR = vec4(0.45, 0.85, 1.0, 1);
vec4 SUN_COLOR = vec4(1, 1, 1, 1);
float SKY_POWER = 0.7;
float SUN_POWER = 7;
Vector3 SUN_DIRECTION = Vector3(0.05, 1.0, 0.05); //TODO calculate actual sun angle, use THIS to compare to the reflected ray


class Material {
public:
    vec4 specular, ambient, diffuse, n, k;
    vec4 reflection;

    vec4 fresnel(Vector3 normal, Vector3 direction) {
        float cosA = (float) fabs(normal * direction.normalize());
        vec4 f0 = ((n - vec4(1, 1, 1, 1))*(n - vec4(1, 1, 1, 1)) + k*k) / ((n + vec4(1, 1, 1, 1))*(n + vec4(1, 1, 1, 1)) + k*k);
        return  f0 + ((vec4(1, 1, 1, 1)) - f0) * pow(1.0 - cosA, 5);
    }

};

class Ray {
public:
    Vector3 origin, direction;
    Ray() {
    }
    Ray(Vector3 origin, Vector3 direction) {
        this->origin = origin;
        this->direction = direction;
    }
};

struct Hit {
    float distance;
    Vector3 position;
    Vector3 direction;
    Vector3 surfaceNormal;
    Material* material;
    Hit() { distance = -1; }
    Ray getRay() {
        return Ray(position, direction);
    }
};

// handle of the shader program
unsigned int shaderProgram;

class Camera
{
public:
    Vector3 eye, lookat, right, up;

    Camera() {}

    Ray getRay(int x, int y){

        Vector3 point = lookat + (right * (2.0 * (x + 0.5) / 600.0 - 1) + up * (2.0 * (y + 0.5) / 600.0 - 1));
        Ray r;
        r.direction = (point - eye).normalize();
        r.origin = eye;

        return r;
    }
};

class WorldObject {
public:
    Material material;
	virtual Hit intersect(Ray ray) = 0;
};

struct Ball: public WorldObject{

    float radius;
    Vector3 origo;

    Ball(float radius, Vector3 origo, bool silver) {
        material.reflection = vec4(1, 1, 1, 1);
        material.n = vec4(0.17, 0.35, 1.5);
        material.k = vec4(3.1, 2.7, 1.9);

        if (silver) {
            material.n = vec4(0.14, 0.16, 0.13);
            material.k = vec4(4.1, 2.3, 3.1);
        }


        this->radius = radius;
        this->origo = origo;
    }

    Hit intersect(Ray ray){
        Hit result;
        result.material = &material;

        float a =   ((ray.direction.x*ray.direction.x)) +
                    ((ray.direction.y*ray.direction.y)) +
                    ((ray.direction.z*ray.direction.z));

        float b = (float) (((2.0 * (ray.origin.x - origo.x) * ray.direction.x)) +
                           ((2.0*(ray.origin.y-origo.y)*ray.direction.y)) +
                           ((2.0*(ray.origin.z-origo.z)*ray.direction.z)));

        float c=    (((ray.origin.x-origo.x)*(ray.origin.x-origo.x))) +
                    (((ray.origin.y-origo.y)*(ray.origin.y-origo.y))) +
                    (((ray.origin.z-origo.z)*(ray.origin.z-origo.z))) - radius*radius;

        float distance = (float) (pow(b, 2.0) - 4.0 * a * c);


        if(distance < 0) {
            result.distance = -1.0f;
            return result;
        }

        float distance1 = (float) ((-b + sqrtf(distance)) / (2.0 * a));
        float distance2 = (float) ((-b - sqrtf(distance)) / (2.0 * a));

//        Vector3 intersection = ray.origin + ray.direction * distance1;
//        if(distance1 > EPSILON && (distance1 < distance2 || distance2 < EPSILON) ) {
//
//            result.position = intersection;
//            result.direction = (intersection - origo).normalize();
//            result.distance = distance1;
//
//            return result;
//        }

        Vector3 R2 = ray.origin + ray.direction * distance2;
        if(distance2 > EPSILON && (distance2 < distance1 || distance1 < EPSILON)) {

            result.position = R2;
            result.surfaceNormal = (R2 - origo).normalize();
            result.direction = ray.direction + (result.surfaceNormal * (ray.direction * result.surfaceNormal) * -2.0f);
            result.distance = distance2;

            return result;
        }

        result.distance = -1;
        return result;
    }

};

class Ground: public WorldObject {
public:
    Vector3 origin, normal;

    Ground(Vector3 o, Vector3 n):origin(o), normal(n) {
        material = Material();
        material.ambient = vec4(0, 0.3, 0, 1);
        material.diffuse = vec4(0.4, 0.6, 0.2, 1);
        material.reflection = 0.0f;
    }

    Hit intersect(Ray ray){
        //I used help from http://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector

        normalize();
        Hit result;

        result.material = &material;

        float cosAngle = normal * ray.direction;
        if(cosAngle == 0) {
            return result;
        }
        float distance = (float) ((normal * (ray.origin - origin)) * (1.0 / cosAngle) * (-1.0));

        if(distance > EPSILON){
            result.position = ray.origin + ray.direction * distance;
            result.direction = ray.direction + (normal * (ray.direction * normal) * -2.0f);
            result.distance = distance;
        }

        return result;
    }

    void normalize() {
        normal = normal.normalize();
    }
};

class Triangle : public WorldObject{
public:
 Vector3 r1, r2, r3;
 Vector3 normal;

    Triangle(Vector3 r1, Vector3 r2, Vector3 r3): r1(r1), r2(r2), r3(r3) {
        material = Material();
        material.ambient = vec4(0, 0.1, 0, 4);
        material.diffuse = vec4(0.1, 0.2, 0.6, 1);
        material.reflection = 0.0f;
        this->normal = ((r2-r1)%(r3-r1)).normalize();
    }
Hit intersect(Ray ray){
        //I used help from http://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector

        normalize();
        Hit result;

        result.material = &material;     

        float cosAngle = normal * ray.direction;
        if(cosAngle == 0) {
            return result;
        }
        
        float distance = (float) ((normal * (ray.origin - this->r1)) * (1.0 / cosAngle) * (-1.0));

        if(distance > EPSILON){
            result.position = ray.origin + ray.direction * distance;            
            result.direction = ray.direction + (normal * (ray.direction * normal) * -2.0f);
            result.distance = distance;            
            
            if(((r2 - r1) % (result.position - r1)) * normal > 0) {
                if(((r3 - r2) % (result.position - r2)) * normal > 0) {
                    if(((r1 - r3) % (result.position - r3)) * normal > 0) {
						return result;
                     }
                     else {
						return Hit();
					 }
                }
           }
		}  
}   
  void normalize() {
      normal = normal.normalize(); 
}
    
};

#define NUMOBJECTS 13

class Scene {
public:
    WorldObject* objects[NUMOBJECTS];
    Scene() {
		Vector3 A(50.0, 0.01, 30.0);
		Vector3 B(50.0, 30.01, 30.0);
		Vector3 C(50.0, 0.01, 130.0);
		Vector3 D(50.0, 30.01, 130.0);
		Vector3 E(550.0, 0.01, 130.0);
		Vector3 F(550.0, 30.01, 130.0);
		Vector3 G(550.0, 0.01, 30.0);
		Vector3 H(550.0, 30.01, 30.0);
		
		
        objects[0] = new Ground(Vector3(0, 0, 0), Vector3(0, 1, 0));
        objects[1] = new Ball(20, Vector3(150, 30, 80), false);
        objects[2] = new Ball(20, Vector3(450, 30, 80), true);
                
        objects[3] = new Triangle(A,B,C);
        objects[4] = new Triangle(B,D,C);
        objects[5] = new Triangle(D,C,E);
        objects[6] = new Triangle(F,D,E);
        objects[7] = new Triangle(F,E,G);
        objects[8] = new Triangle(G,H,F);
        objects[9] = new Triangle(G,E,A);
        objects[10] = new Triangle(C,E,A);
        objects[11] = new Triangle(B,H,A);
        objects[12] = new Triangle(A,H,G);
    }
};

class Raytracer {
public:
    Scene scene;
    Camera camera;

    Raytracer() {
        camera.eye = Vector3(500, 300, 1100);
        camera.lookat = Vector3(270,130,70);
        camera.up = Vector3(0,250,0);
        camera.right = Vector3(250,0,0);
    }

    void render(vec4 frameBuffer[]) {
        for (int x = 0; x < 600; x++) {
            for (int y = 0; y < 600; y++) {
                Ray ray = camera.getRay(x, y);
                vec4 color = raytrace(ray, 0);
                frameBuffer[y * 600 + x] = color;
            }
        }
    }

    vec4 raytrace(Ray ray, int depth) {
        if (depth > 50) {
            return vec4();
        }
        Hit hit = firstIntersect(ray);
        if (hit.distance > 0) {
            return getColor(hit) + raytrace(hit.getRay(), depth + 1) * hit.material->reflection * hit.material->fresnel(hit.direction, hit.direction);
        } else {
            vec4 skyColor = depth == 0 ? SKY_BACKGROUND_COLOR : SKY_COLOR * SKY_POWER;

            if (ray.direction.t() <  1.50 && ray.direction.t() > 1.0) {
                if (ray.direction.p() > 1.0 && ray.direction.p() < 1.5) {
                    skyColor = SUN_COLOR;
                }
            }
            return skyColor;
        }
    }

    vec4 getColor(Hit hit) {
        vec4 color;
        if (hit.distance > 0) {
            if (lineToSun(hit.position)) {
                color = hit.material->diffuse; //MULTIPLY WITH ANGLE
            }
            return color + hit.material->ambient;
        }
        return color;
    }

    bool lineToSun(Vector3 origin) {
        Ray ray = Ray(origin, SUN_DIRECTION);
        Hit hit = firstIntersect(ray);
        return hit.distance < 0;
    }

    Hit firstIntersect(Ray ray) {
        Hit hit;
        for (int i=0; i < NUMOBJECTS; i++) {
            Hit aHit = scene.objects[i]->intersect(ray);
            if (aHit.distance > 0 && (hit.distance < 0 || aHit.distance < hit.distance)) {
                hit = aHit;
            }
        }
        return hit;
    }

};





class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec4 image[windowWidth * windowHeight]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		static float vertexCoords[] = { -1, -1,   1, -1,  -1, 1,
			                             1, -1,   1,  1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,	0, NULL);     // stride and offset: it is tightly packed

		// Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
	}
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	vec4 background[windowWidth * windowHeight];
    Raytracer scene;
    scene.render(background);



	fullScreenTexturedQuad.Create( background );

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
