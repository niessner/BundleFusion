# include "desktop-server.h"
# include "desktop-ui.h"
# include "../dependencies/headers/glfw3.h"

namespace uplink {

//------------------------------------------------------------------------------

template <typename T>
inline T
keepInRange (T value, T minValue, T maxValue)
{
    if (value < minValue)
        return minValue;

    if (value > maxValue)
        return maxValue;
    
    return value;
}

inline void
convertDepthToInverseDepthRgba (
    const uint16* depthValues,
    int numPixels,
    float scaleToMeters,
    uint8* rgbaBuffer,
    uint8 noDepthAlpha = 0,
    uint8 depthAlpha = 255
)
{
    for (int i = 0; i < numPixels; ++i)
    {
        int r = 0, g = 0, b = 0;

        uint8 alpha = noDepthAlpha;

        uint16 depth = depthValues[i];

        if (0 < depth && depth < shift2depth(0xffff))
        {
            const float idepth = 1.f / (float(depth) * scaleToMeters);

            // rainbow between 0 and 4
            r = int((0.f - idepth) * 255.f / 1.f);
            g = int((1.f - idepth) * 255.f / 1.f);
            b = int((2.f - idepth) * 255.f / 1.f);

            if (r < 0)
                r = -r;
            
            if (g < 0)
                g = -g;
            
            if (b < 0)
                b = -b;

            alpha = depthAlpha;
        }

        uint8_t rc = keepInRange (r, 0, 255);
        uint8_t gc = keepInRange (g, 0, 255);
        uint8_t bc = keepInRange (b, 0, 255);

        rgbaBuffer[4 * i    ] = 255 - rc;
        rgbaBuffer[4 * i + 1] = 255 - gc;
        rgbaBuffer[4 * i + 2] = 255 - bc;
        rgbaBuffer[4 * i + 3] = alpha;
    }
}

//------------------------------------------------------------------------------

struct DesktopServerUI::Impl
{
    typedef DesktopServerUI That;

    static const int initialWindowWidth = 640; 
    static const int initialWindowHeight = 480;
    int _width , _height;

    static Impl* impl(GLFWwindow* window)
    {
        return reinterpret_cast<Impl*>(glfwGetWindowUserPointer(window));
    }

    static void resize_callback(GLFWwindow* window, int width, int height)
    {
        glfwMakeContextCurrent(window);

        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glfwMakeContextCurrent(0);
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            switch (key)
            {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            default:
                break;
            }
        }
    }

    Impl(That* that)
    : that(that)
    , mainWindow(0)
    , sharingWindow(0)
    {
        if (!glfwInit())
        {
            uplink_log_error("Failed to initialize GLFW.");
            abort();
        }

        mainWindow = glfwCreateWindow(
            initialWindowWidth,
            initialWindowHeight,
            "Uplink",
            0,
            0
        );

        // Create a sharing context with a hidden window in order to allow other threads to upload data independently from the rendering thread.
        sharingWindow = glfwCreateWindow(
            1,
            1,
            "",
            0,
            mainWindow
        );
        glfwHideWindow(sharingWindow);

        glfwSetWindowUserPointer(mainWindow, this);

        glfwSetWindowSizeCallback(mainWindow, resize_callback);
        glfwSetKeyCallback(mainWindow, key_callback);
        glfwSwapInterval(1);
        glfwSetTime(0.0);

        glfwMakeContextCurrent(mainWindow);
     
        glGenTextures(1, &colorTextureId);
        glGenTextures(1, &depthTextureId);
     
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        glEnable(GL_TEXTURE_2D);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glClearColor(0., 0., 0., 0.);

        glfwMakeContextCurrent(0);
    }

    ~Impl()
    {
        if (0 != mainWindow)
        {
            glfwDestroyWindow(mainWindow);
            mainWindow = 0;            
        }

        if (0 != sharingWindow)
        {
            glfwDestroyWindow(sharingWindow);
            sharingWindow = 0;
        }

        glfwTerminate();
    }

    void glDraw (GLFWwindow* window, double t)
    {
        static const GLfloat vertices [] = {
             -1, -1,  0,
             -1, 1,  0,
             1, 1,  0,
             1, -1,  0
        };

        glVertexPointer(3, GL_FLOAT, 0, vertices);

        static const GLfloat textureCoordinates [] = {
            0, 1,
            0, 0,
            1, 0,
            1, 1

        };

        glTexCoordPointer(2, GL_FLOAT, 0, textureCoordinates);

        static const GLubyte indices[] = {
            0, 1, 2, 3
        };

        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, colorTextureId);
        glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices);

        glBindTexture(GL_TEXTURE_2D, depthTextureId);
        glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices);
    }

    void run ()
    {
        while (!glfwWindowShouldClose(mainWindow))
        {
            {
                glfwMakeContextCurrent(mainWindow);

                glDraw(mainWindow, glfwGetTime());

                glFlush();

                // NOTE: On windows, swapping buffers can block a lot, regardless of the swap interval.
                // NOTE: On windows, the current buffer swapping implementation doesn't require a current context. 
                glfwSwapBuffers(mainWindow);

                glfwMakeContextCurrent(0);
            }

            glfwPollEvents();
        }
    }

    That* that;

    GLFWwindow* mainWindow;

    GLFWwindow* sharingWindow;

    GLuint depthTextureId;
    GLuint colorTextureId;

    std::vector<uint8> renderedDepth;

    Mutex sharingMutex;
};

inline
DesktopServerUI::DesktopServerUI()
: impl(new Impl(this))
{
    const uint8 blankColor [] = { 0, 0, 0, 0 };
    setColorImage(blankColor, 1, 1);

    const uint16 blankDepth [] = { 0 };
    setDepthImage(blankDepth, 1, 1);
}

inline
DesktopServerUI::~DesktopServerUI()
{

}

inline void
DesktopServerUI::setColorImage (const uint8* buffer, int width, int height)
{
    if (0 == impl->mainWindow || 0 == impl->sharingWindow)
        return;

    const MutexLocker _(impl->sharingMutex);

    glfwMakeContextCurrent(impl->sharingWindow);

    glBindTexture(GL_TEXTURE_2D, impl->colorTextureId);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    
    // FIXME: Use glTexSubImage2D when size didn't change.
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        buffer
    );

    glFlush();

    glfwMakeContextCurrent(0);
}

inline void
DesktopServerUI::setDepthImage (const uint16* buffer, int width, int height)
{
    if (0 == impl->mainWindow || 0 == impl->sharingWindow)
        return;

    static int count = 0;

    if (0 == count++ % 30)
        uplink_log_info("Central depth: %d", buffer[(height / 2) * width + width / 2]);

    impl->renderedDepth.resize(width * height * 4);

    convertDepthToInverseDepthRgba(buffer, width * height, 1e-3f, impl->renderedDepth.data(), 0, 127);

    {
        const MutexLocker _(impl->sharingMutex);

        glfwMakeContextCurrent(impl->sharingWindow);

        glBindTexture(GL_TEXTURE_2D, impl->depthTextureId);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        // FIXME: Use glTexSubImage2D when size didn't change.
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            impl->renderedDepth.data()
        );

        glFlush();

        glfwMakeContextCurrent(0);
    }
}

inline void
DesktopServerUI::run()
{
    impl->run();
}

} // uplink namespace
