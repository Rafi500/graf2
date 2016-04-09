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


struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}
};


// handle of the shader program
unsigned int shaderProgram;
//
//class FullScreenTexturedQuad {
//	unsigned int vao, textureId;	// vertex array object id and texture id
//public:
//	void Create(vec4 image[windowWidth * windowHeight]) {
//		glGenVertexArrays(1, &vao);	// create 1 vertex array object
//		glBindVertexArray(vao);		// make it active
//
//		unsigned int vbo;		// vertex buffer objects
//		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects
//
//								// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
//		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
//		static float vertexCoords[] = { -1, -1,   1, -1,  -1, 1,
//			1, -1,   1,  1,  -1, 1 };	// two triangles forming a quad
//		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
//																							   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
//		glEnableVertexAttribArray(0);
//		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
//
//																	  // Create objects by setting up their vertex data on the GPU
//		glGenTextures(1, &textureId);  				// id generation
//		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
//
//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, image); // To GPU
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	}
//
//	void Draw() {
//		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
//		int location = glGetUniformLocation(shaderProgram, "textureUnit");
//		if (location >= 0) {
//			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
//			glActiveTexture(GL_TEXTURE0);
//			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
//		}
//		//glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
//		glDrawPixels(windowWidth, windowHeight, GL_RGB, GL_FLOAT, background);
//	}
//};

 //The virtual world: single quad
//FullScreenTexturedQuad fullScreenTexturedQuad;





//int xszorzo = 15;
//int yszorzo = 25;



//--------------------------------------------------------
// 3D Vektor
//--------------------------------------------------------
struct Vector {
	float x, y, z;

	Vector() {
		x = y = z = 0;
	}
	Vector(float x0, float y0, float z0 = 0) {
		x = x0; y = y0; z = z0;
	}
	Vector operator*(float a) {
		return Vector(x * a, y * a, z * a);
	}
	Vector operator/(float a) {
		return Vector(x / a, y / a, z / a);
	}
	Vector operator+(const Vector v) {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	Vector operator-(const Vector v) {
		return Vector(x - v.x, y - v.y, z - v.z);
	}
	Vector normalizalt()
	{
		Vector temp(x, y, z);
		temp = temp * (1.0 / sqrt(x*x + y*y + z*z));
		return temp;
	}
	float operator*(Vector v) {     // dot product
		return (x * v.x + y * v.y + z * v.z);
	}
	Vector operator%(const Vector v) {     // cross product
		return Vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}
	float tavolsag(Vector v)
	{
		float temp;
		temp = pow(x - v.x, 2) + pow(y - v.y, 2) + pow(z - v.z, 2);
		return sqrt(temp);
	}
	float Length() { return sqrt(x * x + y * y + z * z); }
};



//--------------------------------------------------------
// Spektrum illetve szin
//--------------------------------------------------------
struct Color {
	float r, g, b;

	Color() {
		r = g = b = 0;
	}
	Color(float r0, float g0, float b0) {
		r = r0; g = g0; b = b0;
	}
	Color operator-(const Color& c)
	{
		return Color(r - c.r, g - c.g, b - c.b);
	}
	Color operator*(float a) {
		return Color(r * a, g * a, b * a);
	}
	Color operator*(const Color& c) {
		return Color(r * c.r, g * c.g, b * c.b);
	}
	Color operator+(const Color& c) {
		return Color(r + c.r, g + c.g, b + c.b);
	}
	Color operator/(const Color& c) {
		return Color(r / c.r, g / c.g, b / c.b);
	}
};
static Color background[windowWidth * windowHeight];
struct Ray
{
	Vector kpont;
	Vector irany;

	Ray(Vector k = Vector(0, 0, 0), Vector i = Vector(0, 0, 0))
	{
		kpont = k; irany = i;
	}

	Vector getPos(float t)
	{
		return (kpont + irany * t);
	}
};

class Camera
{
public:
	Vector eye, lookat, up, right;

	Camera()
	{
		Vector v(0, 0, 0);
		eye = lookat = up = right = v;
	}
	Ray Getray(float x, float y)
	{
		Vector p = lookat + right * (2.0*x / (600.0) - 1.0) + up * (1.0 - (2.0 * y / (600.0))) + Vector(0.5, 0.5, 0.5);
		Ray r(eye, p - eye);
		return r;
	}
};

class Material
{
public:
	bool isReflective;
	bool isRefractive;
	Color f0;
	Color kd, ks, ka;
	float shininess;
	Color k;
	Color n;

	Material(bool l = false, bool r = false, Color kk = Color(0, 0, 0), Color nn = Color(0, 0, 0))
	{
		isReflective = l;
		isRefractive = r;
		k = kk;
		n = nn;
		f0 = ((n - Color(1, 1, 1))*(n - Color(1, 1, 1)) + k*k) / ((n + Color(1, 1, 1))*(n + Color(1, 1, 1)) + k*k);
	}

	void szamol()
	{
		f0 = ((n - Color(1, 1, 1))*(n - Color(1, 1, 1)) + k*k) / ((n + Color(1, 1, 1))*(n + Color(1, 1, 1)) + k*k);
	}

	Color Fresnel(Vector N, Vector V) {
		float cosa = fabs(N * V.normalizalt());
		f0 = ((n - Color(1, 1, 1))*(n - Color(1, 1, 1)) + k*k) / ((n + Color(1, 1, 1))*(n + Color(1, 1, 1)) + k*k);
		return  f0 + ((Color(1, 1, 1)) - f0) * pow(1.0 - cosa, 5);
	}

	Color  ReflectedRadiance(Vector L, Vector N, Vector V, Color Lin) {
		float costheta = N * L.normalizalt();
		if (costheta < 0) return Color(0, 0, 0);
		Color Lref = Lin * kd * costheta;
		Vector H = L + V;
		Vector hnorm = H.normalizalt();
		float cosdelta = N * hnorm;
		if (cosdelta < 0) return Lref;
		Lref = Lref +
			Lin * ks * pow(cosdelta, shininess);
		return Lref;
	}


};

class Light
{
public:
	Color power;
	Vector poz;

	Light(Color pov = Color(0, 0, 0), Vector po = Vector(0, 0, 0))
	{
		power = pov; poz = po;
	}

	Vector getDir(Vector p0)
	{
		Vector dir;
		dir = poz - p0;
		dir = dir.normalizalt();
		return dir;
	}

	float gettav(Vector p0)
	{
		float tav;
		float temp;
		temp = pow((poz.x - p0.x), 2) + pow((poz.y - p0.y), 2) + pow((poz.z - p0.z), 2);
		tav = sqrt(temp);
		return tav;
	}
};

class Henger
{
public:
	Material mal;
	Vector p0;
	float R;
	Vector a;
	Vector p1;

	Henger(Vector pp0 = Vector(0, 0, 0), float Rr = 0, Vector aa = Vector(0, 0, 0), Vector pp1 = Vector(0, 0, 0))
	{
		p0 = pp0;
		R = Rr;
		a = aa;
		p1 = pp1;
	}

	float Intersect(Ray ray)
	{
		float t = -1;

		float A = (ray.irany - a*(ray.irany*a))*((ray.irany - a*(ray.irany*a)));
		float B = 2.0 * ((ray.irany - a*(ray.irany*a)) * ((ray.kpont - p0) - a*((ray.kpont - p0)*a)));
		float C = ((ray.kpont - p0) - a*((ray.kpont - p0)*a)) * ((ray.kpont - p0) - a*((ray.kpont - p0)*a)) - R*R;

		float D = B*B - 4.0 * A * C;
		if (D < 0) return -1;
		float t1 = (-1.0 * B - sqrt(D)) / (2.0 * A);
		float t2 = (-1.0 * B + sqrt(D)) / (2.0 * A);

		if (t2>0.0001) t = t2;
		if (t1>0.0001 && t1<t2) t = t1;

		if ((((a.normalizalt()*((ray.kpont + (ray.irany * t)) - p0).normalizalt()) < 0)) && ((a.normalizalt()*((ray.kpont + (ray.irany * t2)) - p0).normalizalt()) >= 0)) t = t2;
		if ((((a.normalizalt()*((ray.kpont + (ray.irany * t)) - p0).normalizalt()) < 0)) && ((a.normalizalt()*((ray.kpont + (ray.irany * t1)) - p0).normalizalt()) >= 0)) t = t1;

		if ((((a.normalizalt()*((ray.kpont + (ray.irany * t)) - p1).normalizalt()) > 0.0001)) && ((a.normalizalt()*((ray.kpont + (ray.irany * t2)) - p1).normalizalt()) <= 0)) t = t2;
		if ((((a.normalizalt()*((ray.kpont + (ray.irany * t)) - p1).normalizalt()) > 0.0001)) && ((a.normalizalt()*((ray.kpont + (ray.irany * t1)) - p1).normalizalt()) <= 0)) t = t1;

		if ((t > 0.0001) && (((a.normalizalt()*((ray.kpont + (ray.irany * t)) - p0).normalizalt()) >= 0)) && (((a.normalizalt()*((ray.kpont + (ray.irany * t)) - p1).normalizalt()) <= 0))) return t;
		else return -0.1;
	}

	Vector getNorm(Vector p)
	{
		return (((p - p0) - (a * (a *(p - p0)))) / 2).normalizalt();
	}

	Vector Reflecteddir(Vector N, Vector V)
	{
		float cosa = N*V * (-1.0);
		Vector R;
		R = V + N * cosa * 2.0;
		return R;
	}

	bool RefractionDir(Vector& T, Vector N, Vector V) {
		float cosa = N * V * (-1.0);
		float cn = mal.n.r;
		Vector Ne = N;
		if (cosa < 0)
		{
			cosa = -cosa; Ne = N * (-1.0); cn = mal.n.r*(0.1);
		}
		float disc = 1.0 - (1.0 - cosa * cosa) / cn / cn;
		if (disc < 0) return false;
		T = V / cn + Ne * (cosa / cn - sqrt(disc));
		return true;
	}


};

class Paraboloid
{
public:
	Material mal;
	Vector p0;
	Vector n;
	float f;
	float maxf;

	float Intersect(Ray ray)
	{
		float t = -1;
		Vector v = ray.irany;
		Vector eye = ray.kpont;
		float A = v*v - (n*v)*(n*v);
		float B = 2.0 * (v * eye + (p0*n)*(v*n) - v*(n*f) - v*p0 - (eye*n)*(v*n));
		float C = p0*p0 + 2.0*(p0*(n*f)) + (n*f)*(n*f) - 2.0*(p0*eye) - 2.0*((n*f)*eye) + eye*eye - (n*eye)*(n*eye) + 2.0*(n*eye)*(n*p0) - (n*p0)*(n*p0);

		float D = B*B - 4.0*A*C;
		if (D < 0) return -1;
		float t1 = (-1.0 * B - sqrt(D)) / (2.0 * A);
		float t2 = (-1.0 * B + sqrt(D)) / (2.0 * A);


		if (t2>0.0001) t = t2;
		if (t1>0.0001 && t1<t2) t = t1;

		if (((((n.normalizalt())*((ray.kpont + (ray.irany * t)) - (p0 + n*maxf)).normalizalt()) > 0.0001)) && ((((n.normalizalt())*((ray.kpont + (ray.irany * t2)) - (p0 + n*maxf)).normalizalt()) < 0))) t = t2;
		if (((((n.normalizalt())*((ray.kpont + (ray.irany * t)) - (p0 + n*maxf)).normalizalt()) > 0.0001)) && ((((n.normalizalt())*((ray.kpont + (ray.irany * t1)) - (p0 + n*maxf)).normalizalt()) < 0))) t = t1;

		if ((t > 0.0001) && (((n.normalizalt()*((ray.kpont + (ray.irany * t)) - (p0 + n*maxf)).normalizalt()) < 0))) return t;
		else return -0.1;
	}

	Vector Reflecteddir(Vector N, Vector V)
	{
		float cosa = N*V * (-1.0);
		Vector R;
		R = V + N * cosa * 2.0;
		return R;
	}

	bool RefractionDir(Vector& T, Vector N, Vector V) {
		float cosa = N * V * (-1.0);
		float cn = mal.n.r;
		Vector Ne = N;
		if (cosa < 0)
		{
			cosa = -cosa; Ne = N * (-1.0); cn = mal.n.r*(0.1);
		}
		float disc = 1.0 - (1.0 - cosa * cosa) / cn / cn;
		if (disc < 0) return false;
		T = V / cn + Ne * (cosa / cn - sqrt(disc));
		return true;
	}


	Vector getNorm(Vector p)
	{
		return ((p0 + n*f - p) - (n * (n *(p - p0))) * 2).normalizalt();
	}

};

class Sik
{
public:
	Vector a, b, c, d;
	Vector irany, normal;
	Material malA;

	Sik(){}

	Sik(Vector normalv, Vector x, Vector y, Vector z, Vector w) {
		normal = normalv;
		a = x;
		b = y;
		c = z;
		d = w;
	}

	float Intersect(Ray ray)
	{
		float t = -1;
		t = (ray.kpont.y / ray.irany.y) * (-1.0);
		//if (t<0) return -1;
		//if (t>0.00001) return t;
		if (t<1.0) return -1;
		if (t>0.00001) return t;
		else return -1;
	}

	Vector getNorm(Vector p)
	{
		//return Vector(0, 1, 0);
		return normal;
	}

	Vector Reflecteddir(Vector N, Vector V)
	{
		float cosa = N*V * (-1.0);
		Vector R;
		R = V + N * cosa * 2.0;
		return R;
	}

	bool RefractionDir(Vector& T, Vector N, Vector V) {
		float cosa = N * V * (-1.0);
		float cn = malA.n.r;
		Vector Ne = N;
		if (cosa < 0)
		{
			cosa = -cosa; Ne = N * (-1.0); cn = malA.n.r*(0.1);
		}
		float disc = 1.0 - (1.0 - cosa * cosa) / cn / cn;
		if (disc < 0) return false;
		T = V / cn + Ne * (cosa / cn - sqrt(disc));
		return true;
	}
};

//struct Foton
//{
//	Color power;
//	Vector talalat;
//	Vector beirany;
//	Vector norm;
//
//};

//Color kep[360000];
class Scene
{
public:
	Camera camera;
	Henger objects_henger[100];
	int ocount_henger;
	Paraboloid objects_para[100];
	int ocount_para;
	Light lights[100];
	int lcount;
	Color La;
	Sik sik;
	Sik masik;
	

	Scene()
	{
		lcount = 0; ocount_henger = 0;
		sik = Sik(Vector(0, 1, 0), Vector(0, 1, 0), Vector(0, 1, 0), Vector(0, 1, 0), Vector(0, 1, 0));
	}

	void addo(Henger object)
	{
		objects_henger[ocount_henger] = object;
		ocount_henger++;
	}

	void addp(Paraboloid object)
	{
		objects_para[ocount_para] = object;
		ocount_para++;
	}

	void addl(Light light)
	{
		lights[lcount] = light;
		lcount++;
	}


	void Anyag_ezust(Material &mal)
	{
		mal.isReflective = true;
		mal.isRefractive = false;
		mal.n = Color(0.14, 0.16, 0.13);
		mal.k = Color(4.1, 2.3, 3.1);
		mal.ka = Color(0.19225, 0.19225, 0.19225);
		mal.shininess = 0.7;
	}

	void Anyag_uveg(Material &mal)
	{
		mal.isReflective = true;
		mal.isRefractive = false;
		mal.n = Color(1.5, 1.5, 1.5);
		mal.k = Color(0, 0, 0);
		mal.ka = Color(0.042, 0.142, 0.242);
		mal.shininess = 0.7;
	}

	void Anyag_beall(Material &mal, int tp)
	{
		if (tp == 1)
			Anyag_uveg(mal);
		if (tp == 2)
			Anyag_ezust(mal);
	}

	void Kaktusz_parab()
	{
		int kx = 300;
		int ky = 0;
		int kz = 100;
		int tp = 2;


		Paraboloid torzs;
		torzs.p0 = Vector(kx, ky + 0, kz + 150);
		torzs.f = 5;
		torzs.maxf = 355;
		torzs.n = Vector(0, 1, 0).normalizalt();

		torzs.mal.szamol();
		Anyag_beall(torzs.mal, tp);
		addp(torzs);

		//srand((unsigned)(glutGet(GLUT_ELAPSED_TIME)*xszorzo));
		//int isz = 90;// -rand() % 180;
		//float sz = isz*3.14 / 180;
		//float nx = sin(sz);
		//float nz = -cos(sz);
		//int r = 140;

		//Paraboloid ag_1;
		//ag_1.p0 = Vector(kx, ky + 120, kz + 150);
		//ag_1.f = 2;
		//ag_1.maxf = 150;
		//ag_1.n = Vector(-1 * nx, 0, nz).normalizalt();

		//ag_1.mal.szamol();
		//Anyag_beall(ag_1.mal, tp);
		//addp(ag_1);


		//Paraboloid ag_2;
		//ag_2.p0 = Vector(kx - r*nx, ky + 120, kz + 150 + r*nz);
		//ag_2.f = 1;
		//ag_2.maxf = 80;
		//ag_2.n = Vector(0, 1, 0).normalizalt();

		//ag_2.mal.szamol();
		//Anyag_beall(ag_2.mal, tp);
		//addp(ag_2);
	}

	void Kaktusz_henger()
	{
		int kx = 500;
		int ky = 0;
		int kz = 100;
		int tp = 1;


		Henger torzs;
		torzs.p0 = Vector(kx, ky + 0, kz + 200);
		torzs.R = 50;
		torzs.p1 = Vector(kx, ky + 400, kz + 200);
		torzs.a = (torzs.p1 - torzs.p0).normalizalt();

		torzs.mal.szamol();
		Anyag_beall(torzs.mal, tp);
		addo(torzs);


		//srand((unsigned)(glutGet(GLUT_ELAPSED_TIME)*yszorzo));
		//int isz = 90;// -rand() % 180;
		//float sz = isz*3.14 / 180;
		//float nx = sin(sz);
		//float nz = -cos(sz);
		//int r1 = 140;
		//int r2 = 120;

		/*Henger ag_1;
		ag_1.p0 = Vector(kx, ky + 250, kz + 200);
		ag_1.R = 25;
		ag_1.p1 = Vector(kx + r1*nx, ky + 250, kz + 200 + r1*nz);
		ag_1.a = (ag_1.p1 - ag_1.p0).normalizalt();

		ag_1.mal.szamol();
		Anyag_beall(ag_1.mal, tp);
		addo(ag_1);

		Henger ag_2;
		ag_2.p0 = Vector(kx + r2*nx, ky + 250, kz + 200 + r2*nz);
		ag_2.R = 15;
		ag_2.p1 = Vector(kx + r2*nx, ky + 350, kz + 200 + r2*nz);
		ag_2.a = (ag_2.p1 - ag_2.p0).normalizalt();

		ag_2.mal.szamol();
		Anyag_beall(ag_2.mal, tp);
		addo(ag_2);*/
	}

	void feltolt()
	{

		La = Color(1, 1, 1);

		camera.eye = Vector(300, 500, -250);
		camera.lookat = Vector(300, 300, 0);
		camera.right = Vector(300, 0, 0);
		camera.up = Vector(0, 300, 0);

		//------------------------------


		Kaktusz_parab();
		Kaktusz_henger();



		//------------------------------

		Light lr;
		lr.power = Color(0.2, 0.2, 0.2);
		lr.poz = Vector(300, 10000, 0);
		addl(lr);

		/*Light lg;
		lg.power = Color(0.0, 0.1, 0.0);
		lg.poz = Vector(300, 1000, 400);
		addl(lg);

		Light lb;
		lb.power = Color(0.0, 0.0, 0.1);
		lb.poz = Vector(700, 1000, 400);
		addl(lb);*/

		sik.malA.n = Color(0.17, 0.35, 1.5);
		sik.malA.k = Color(3.1, 2.7, 1.9);
		sik.malA.ka = Color(0.24725, 0.1995, 0.0745);
		sik.malA.kd = Color(0.75164, 0.60648, 0.22648);
		sik.malA.ks = Color(0.628281, 0.555802, 0.366065);
		sik.malA.shininess = 0.7;
		sik.malA.isReflective = false;
		sik.malA.isRefractive = true;
		sik.malA.szamol();
	}

	float intersectAll(Ray ray, int& io, int& ip)
	{
		float t = 999999;
		for (int i = 0; i<ocount_para; i++)
		{
			float tnew = objects_para[i].Intersect(ray);
			if ((tnew > 0.0001) && (tnew<t))
			{
				t = tnew;
				ip = i;
			}

		}
		for (int i = 0; i<ocount_henger; i++)
		{
			float tnew = objects_henger[i].Intersect(ray);
			if ((tnew > 0.0001) && (tnew<t))
			{
				t = tnew;
				io = i;
			}

		}
		float tnew = sik.Intersect(ray);
		if ((tnew > 0.0001) && (tnew<t))
		{
			t = tnew;
			io = -1;
		}
		if (t<999999) return t;
		else return (-1);
	}

	Color Trace(Ray ray, int depth)
	{
		if (depth > 5) return La;
		int ind_para = -1;
		int ind_henger = -1;
		float hit = intersectAll(ray, ind_henger, ind_para);

		if (hit < 0) return Color(0.5, 0.6, 0.9);

		Color color(1, 1, 1);

		if (ind_para >= 0) color = objects_para[ind_para].mal.ka * La;
		else if (ind_henger >= 0) color = objects_henger[ind_henger].mal.ka * La;
		else if (ind_henger == -1) color = sik.malA.ka*La;


		Vector x;
		x = ray.kpont + ray.irany * hit;

		//if ((int)(x.x / 50) % 2 == 0 && (int)(x.z / 50) % 2 == 0)
		//{

			//sik.malA.ka = Color(0, 0.44725, 0.44725);
			//sik.malA.kd = Color(0.75164, 0.60648, 0.22648);
			//sik.malA.ks = Color(0.628281, 0.555802, 0.366065);

			sik.malA.ka = Color(0.1, 0.6, 0.2);
			sik.malA.kd = Color(0.1, 0.6, 0.2);
			sik.malA.ks = Color(0.1, 0.6, 0.2);
		/*}
		else
		{
			sik.malA.ka = Color(0.24725, 0.24725, 0);
			sik.malA.kd = Color(0.60648, 0.22648, 0.75164);
			sik.malA.ks = Color(0.555802, 0.366065, 0.628281);
		}*/

			//if ((int)(x.x) > 100 && (int)(x.x) < 500) {
			//	sik.malA.ka = Color(0, 0.1, 1);
			//	sik.malA.kd = Color(0, 0.1, 1);
			//	sik.malA.ks = Color(0, 0.1, 1);
			//}

		Vector N;
		if (ind_para >= 0) N = objects_para[ind_para].getNorm(x);
		else if (ind_henger >= 0) N = objects_henger[ind_henger].getNorm(x);
		else if (ind_henger == -1) N = sik.getNorm(x);

		if ((N.normalizalt()*ray.irany.normalizalt())>0) N = N * (-1.0);
		for (int i = 0; i<lcount; i++)
		{
			Ray shadowRay(x + N*0.01, (lights[i].poz - x).normalizalt());
			int shadowoid_para = 0;
			int shadowoid_henger = 0;
			float shadowHit = intersectAll(shadowRay, shadowoid_henger, shadowoid_para);
			Vector y(0, 0, 0);
			if (shadowHit>0.0001) y = shadowRay.kpont + shadowRay.irany * shadowHit;
			if (shadowHit < 0 || x.tavolsag(y) > x.tavolsag(lights[i].poz))
			{
				Vector V;
				V = ray.irany.normalizalt() * (-1.0);
				Vector Ll;
				Ll = shadowRay.irany.normalizalt();
				Vector Hl;
				Hl = (V + Ll).normalizalt();
				float diffuz = Ll * N;
				Color diff;
				if (diffuz < 0) diff = Color(0, 0, 0);
				else
				{
					if (ind_para >= 0) diff = objects_para[ind_para].mal.kd * diffuz;
					else if (ind_henger >= 0) diff = objects_henger[ind_henger].mal.kd * diffuz;
					else if (ind_henger == -1) diff = sik.malA.kd * diffuz;

				}
				float spekul = Hl * N;
				Color spek;
				if (spekul < 0) spek = Color(0, 0, 0);
				else
				{
					if (ind_para >= 0) spek = objects_para[ind_para].mal.ks * pow(spekul, objects_para[ind_para].mal.shininess);
					else if (ind_henger >= 0) spek = objects_henger[ind_henger].mal.ks * pow(spekul, objects_henger[ind_henger].mal.shininess);
					else if (ind_henger == -1) spek = sik.malA.ks * pow(spekul, sik.malA.shininess);

				}
				color = color + (lights[i].power * (diffuz + spekul));
			}
		}
		if (ind_para >= 0)
		{
			if (objects_para[ind_para].mal.isReflective)
			{
				Vector reflectedIrany;
				reflectedIrany = objects_para[ind_para].Reflecteddir(N, ray.irany.normalizalt());

				Ray reflectedRay(x + N*0.01, reflectedIrany);
				Color fresn;

				color = color + (objects_para[ind_para].mal.Fresnel(N, ray.irany.normalizalt()) * Trace(reflectedRay, depth + 1));
			}
			Vector refractedIrany;
			if (objects_para[ind_para].mal.isRefractive && objects_para[ind_para].RefractionDir(refractedIrany, N, ray.irany.normalizalt()))
			{

				Ray refractedRay(x + N*0.01, refractedIrany);
				color = color + ((Color(1, 1, 1) - objects_para[ind_para].mal.Fresnel(N, ray.irany.normalizalt())) * Trace(refractedRay, depth + 1));
			}
		}
		else if (ind_henger >= 0)
		{
			if (objects_henger[ind_henger].mal.isReflective)
			{
				Vector reflectedIrany;
				reflectedIrany = objects_henger[ind_henger].Reflecteddir(N, ray.irany.normalizalt());

				Ray reflectedRay(x + N*0.01, reflectedIrany);
				Color fresn;

				color = color + (objects_henger[ind_henger].mal.Fresnel(N, ray.irany.normalizalt()) * Trace(reflectedRay, depth + 1));
			}
			Vector refractedIrany;
			if (objects_henger[ind_henger].mal.isRefractive && objects_henger[ind_henger].RefractionDir(refractedIrany, N, ray.irany.normalizalt()))
			{

				Ray refractedRay(x + N*0.01, refractedIrany);
				color = color + ((Color(1, 1, 1) - objects_henger[ind_henger].mal.Fresnel(N, ray.irany.normalizalt())) * Trace(refractedRay, depth + 1));
			}
		}
		else
		{
			if (ind_henger == -1)
			{
				if (sik.malA.isReflective)
				{
					Vector reflectedIrany;
					reflectedIrany = sik.Reflecteddir(sik.getNorm(x), ray.irany.normalizalt());
					Ray reflectedRay(x + N*0.01, reflectedIrany);
					color = color + (sik.malA.Fresnel(sik.getNorm(x), ray.irany.normalizalt()) * Trace(reflectedRay, depth + 1));
				}
				Vector refractedIrany;
				if (sik.malA.isRefractive && sik.RefractionDir(refractedIrany, sik.getNorm(x), ray.irany.normalizalt()))
				{
					Ray refractedRay(x + N*0.01, refractedIrany);
					color = color + ((Color(1, 1, 1) - sik.malA.Fresnel(sik.getNorm(x), ray.irany.normalizalt())) * Trace(refractedRay, depth + 1));
				}
			}
		}
		return color;
	}

	void Render()
	{
		for (int i = 0; i<600; i++)
			for (int j = 0; j<600; j++)
			{
				Ray r = camera.Getray(j, 600 - i);
				Color color = Trace(r, 0);
				//kep[i * 600 + j] = color;
				background[i * 600 + j] = color;

			}
	}
};
Scene scene;




// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	scene.lcount = 0; scene.ocount_henger = 0;


	scene.feltolt();
	scene.Render();

	//static vec4 background[windowWidth * windowHeight];
	/*for (int x = 0; x < windowWidth; x++) {
		for (int y = 0; y < windowHeight; y++) {
			background[y * windowWidth + x] = Vector((float)x / windowWidth, (float)y / windowHeight, 0);
		}
	}*/
	//fullScreenTexturedQuad.Create(background);

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
	glDrawPixels(windowWidth, windowHeight, GL_RGB, GL_FLOAT, background);
	//fullScreenTexturedQuad.Draw();
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
