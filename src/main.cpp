#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <stdio.h>

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/type_ptr.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"
#include <glm/gtc/matrix_transform.hpp>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 mvp;\n"
"void main(){\n"
"gl_Position = mvp * vec4(aPos, 1.0);\n"
"}\n\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"uniform vec2 screenResolution;"
"void main()\n"
"{\n"
"   FragColor = vec4(gl_FragCoord.x / screenResolution.x, gl_FragCoord.y / screenResolution.y, 1, 1.0);\n"
"}\n\0";

const unsigned int SCR_WIDTH = 1080;
const unsigned int SCR_HEIGHT = 720;
unsigned int width = SCR_WIDTH, height = SCR_HEIGHT;

const unsigned int MAX_ITERATIONS = 50500;
const float rho = 28.0f, sigma = 10.0, beta = 8 / 3, dt = 0.003;
std::vector<float> xBuffer, yBuffer, zBuffer;

float distance = 1.0f;
unsigned int numVertices = 0;
glm::mat4 model(1.0f);

void solveDynamicLorenz(int buffer, float sigma, float rho, float beta, float dt) {
    if (numVertices >= MAX_ITERATIONS)
        return;
    // Origin
    static float x = 2, y = 1, z = 1;
    static std::vector<float> vertices;
    // Equations
    float dx = sigma * (y - x) * dt;
    float dy = (x * (rho - z) - y) * dt;
    float dz = (x * y - beta * z) * dt;
    x += dx; y += dy; z += dz;
    // Add to buffers
    vertices.push_back(x);  xBuffer.push_back(x);
    vertices.push_back(y);  yBuffer.push_back(y);
    vertices.push_back(z);  zBuffer.push_back(z);
    // Add to opengl vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    void* ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    memcpy(ptr, &vertices[0], sizeof(float) * vertices.size());
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Increase numVertices
    numVertices++;
}

bool LoadTextureFromFile(const char* filename, GLuint* out_texture, int* out_width, int* out_height) {
    // Load from file
    int image_width = 0;
    int image_height = 0;
    unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);
    if (image_data == NULL)
        return false;

    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    stbi_image_free(image_data);

    *out_texture = image_texture;
    *out_width = image_width;
    *out_height = image_height;

    return true;
}

int main() {

    // glfw
    glfwInit();

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    // Window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LorenzAttractor", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Init glew
    if (glewInit() != GLEW_OK)
        std::cout << "Couldn't initialize GLEW" << std::endl;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    //ImGui::StyleColorsLight();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, MAX_ITERATIONS * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Load equations texture
    int my_image_width = 0;
    int my_image_height = 0;
    GLuint my_image_texture = 0;
    bool ret = LoadTextureFromFile("LorenzSystem.png", &my_image_texture, &my_image_width, &my_image_height);

    // Linear transforms
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), static_cast<float>(SCR_WIDTH / SCR_HEIGHT), 0.001f, 100.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 1), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
    
    model = glm::rotate(model, glm::radians(-10.f), glm::vec3(0, 1.0f, 0));
    model = glm::translate(model, glm::vec3(0.12, -0.11, 0));
    model = glm::scale(model, glm::vec3(0.008));
    
    glm::mat4 mvp = proj * view * model;

    // Enable blending
    glEnable(GL_BLEND | GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Anti aliasing
    glEnable(GL_MULTISAMPLE);

    while (!glfwWindowShouldClose(window)) {

        processInput(window);

        // Solve lorenz system
        static int speed = 1;
        for (int i = 0; i < speed; i++)
            solveDynamicLorenz(VBO, sigma, rho, beta, dt);

        glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Use shader program
        glUseProgram(shaderProgram);

        // View matrix
        glm::mat4 view = glm::lookAt(glm::vec3(0, 0, distance), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));

        // Model matrix
        static float angleX = 0, angleY = 0, angleZ = 0, rotationSpeed = 1.f;
        model = glm::rotate(model, glm::radians(rotationSpeed * angleX), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rotationSpeed * angleY), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rotationSpeed * angleZ), glm::vec3(0.0f, 0.0f, 1.0f));

        // MVP matrix
        glm::mat4 mvp = proj * view * model;
        int location = glGetUniformLocation(shaderProgram, "mvp");
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(mvp));

        // Send screen resolution to fragment shader
        location = glGetUniformLocation(shaderProgram, "screenResolution");
        glUniform2f(location, width, height);

        // Draw
        glBindVertexArray(VAO);
  
        // Draw lines
        static float lineWidth = 2.0f;
        glLineWidth(lineWidth);
        glDrawArrays(GL_LINE_STRIP, 0, numVertices);

        // Draw points
        static float pointSize = 2.0f;
        glPointSize(pointSize);
        glDrawArrays(GL_POINTS, 0, numVertices);

        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Menu bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("Edit")) {
                if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
                if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
                ImGui::Separator();
                if (ImGui::MenuItem("Cut", "CTRL+X")) {}
                if (ImGui::MenuItem("Copy", "CTRL+C")) {}
                if (ImGui::MenuItem("Paste", "CTRL+V")) {}
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        // Lorenz attractor window
        {
            ImGui::Begin("Lorenz Attractor");

            ImGui::BeginGroup();
            ImGui::TextColored(ImColor(200, 0, 255), "Lorenz system:");

            std::string sigmaString = "sigma: " + std::to_string(sigma);
            ImGui::Text(sigmaString.c_str());
            ImGui::SameLine();
            std::string rhoString = "rho: " + std::to_string(rho);
            ImGui::Text(rhoString.c_str());
            ImGui::SameLine();
            std::string betaString = "beta: " + std::to_string(beta);
            ImGui::Text(betaString.c_str());
            ImGui::Image((void*)(intptr_t)my_image_texture, ImVec2(my_image_width, my_image_height));

            ImGui::EndGroup();

            ImGui::SameLine();

            ImGui::BeginGroup();

            ImGui::TextColored(ImColor(200, 0, 255), "Coord plot:");
            ImGui::Text("Value of x, y, z relative to time");
            ImGui::PlotLines("x", &xBuffer[0], xBuffer.size());
            ImGui::PlotLines("y", &yBuffer[0], yBuffer.size());
            ImGui::PlotLines("z", &zBuffer[0], zBuffer.size());

            std::string pointsString = "Num points: " + std::to_string(numVertices);
            ImGui::Text(pointsString.c_str());
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::EndGroup();

            ImGui::End();
        }
        // Animation window
        {
            ImGui::Begin("Animation");
            ImGui::TextColored(ImColor(200, 0, 255), "View:");

            ImGui::SliderFloat("Point size", &pointSize, 0.0f, 10.f);
            ImGui::SliderFloat("Line width", &lineWidth, 0.0f, 10.f);
            ImGui::SliderInt("Speed", &speed, 1.0f, 100.f);

            ImGui::End();
        }
        // Linear transforms window
        {
            ImGui::Begin("Linear transforms");

            ImGui::TextColored(ImColor(200, 0, 255), "Rotation:");
            ImGui::SliderFloat("Rotation speed", &rotationSpeed, 0.f, 5.f);
            ImGui::SliderFloat("Angle x axis", &angleX, -1.f, 1.f);
            ImGui::SliderFloat("Angle y axis", &angleY, -1.f, 1.f);
            ImGui::SliderFloat("Angle z axis", &angleZ, -1.f, 1.f);
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int w, int h) {
    glViewport(0, 0, w, h);
    width = w;
    height = h;
}

void mouse_callback(GLFWwindow* window, int button, int action, int mods) { }

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    static float distanceSpeed = 0.2;
    distance -= yoffset * distanceSpeed;
}