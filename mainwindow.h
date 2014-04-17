#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QWidget>
#include <QString>
#include <QImage>
#include <QVector>

#include "opencv2/imgproc/imgproc.hpp"

class QPushButton;
class QLineEdit;
class QTimer;
class QLabel;
class QTextEdit;

enum NextCmd
{
    CMD_LEFT,
    CMD_RIGHT,
    CMD_UP,
    CMD_DOWN,
    CMD_UNKNOWN
};

class MainWindow : public QWidget
{
    Q_OBJECT

    int cells[4][4];

    QPushButton *runButton;
    QPushButton *nextButton;

    QLineEdit *lineEdit;

    QTextEdit *logArea;

    QTimer *timer;

    QLabel *gameArea;

    QImage getImage(const cv::Mat& mat);

    QVector<int> getHistogram(const cv::Mat mat);

    int getCellNumber(QVector<int> hist);

    NextCmd getNextCommand();

public:
    explicit MainWindow(QWidget *parent = 0);

public slots:
    void runProgram();

    void runCommand();

    void updateGameArea();
};

#endif //MAIN_WINDOW_H
