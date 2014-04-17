#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTimer>
#include <QProcess>
#include <QLabel>
#include <QPainter>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "mainwindow.h"

using namespace cv;

MainWindow::MainWindow(QWidget *parent) : QWidget(parent)
{
    this->setWindowTitle( tr("Game 2048"));

    QVBoxLayout *mainLayout = new QVBoxLayout;

    gameArea = new QLabel;
    QPixmap pixmap(500, 500);
    pixmap.fill(Qt::gray);
    gameArea->setPixmap(pixmap);
    mainLayout->addWidget(gameArea);

    QHBoxLayout *bottomLayout = new QHBoxLayout;
    runButton = new QPushButton("Run");
    connect(runButton, SIGNAL(clicked()), this, SLOT(runProgram()));
    bottomLayout->addWidget(runButton);

    nextButton = new QPushButton("Next");
    connect(nextButton, SIGNAL(clicked()), this, SLOT(runCommand()));
    bottomLayout->addWidget(nextButton);

    lineEdit = new QLineEdit;
    bottomLayout->addWidget(lineEdit);

    mainLayout->addLayout(bottomLayout);

    logArea = new QTextEdit;
    logArea->setReadOnly(true);
    mainLayout->addWidget(logArea);

    setLayout(mainLayout);

    timer = new QTimer;
    connect(timer, SIGNAL(timeout()), this, SLOT(runCommand()));

    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            cells[i][j] = 0;
}

void MainWindow::runProgram()
{
    static bool isRun = false;

    if(!isRun)
    {
        timer->setInterval(1000);
        timer->start();
    }
    else
    {
        timer->stop();
    }
    isRun = !isRun;
}

void MainWindow::runCommand()
{
//    static int delay = 0;

//    if(delay < 7)
//    {
//        delay++;
//        return;
//    }
    updateGameArea();

//    key code 123 -- left arrow Key
//    key code 124 -- right arrow Key
//    key code 126 -- up arrow Key
//    key code 125 -- down arrow Key

    QProcess process;

    int cmdNum;
    switch(getNextCommand())
    {
    case CMD_UP:    cmdNum = 126; break;
    case CMD_DOWN:  cmdNum = 125; break;
    case CMD_LEFT:  cmdNum = 123; break;
    case CMD_RIGHT: cmdNum = 124; break;
    case CMD_UNKNOWN:
    default:
        cmdNum = 124;
    }

    lineEdit->setText(QString::number(cmdNum));

    process.start("osascript", QStringList() << "-e" << "tell application \"System Events\" to key code " + QString::number(cmdNum));
    process.waitForFinished();

    // -e 'tell application "System Events" to key code 123'
}

void MainWindow::updateGameArea()
{
    QProcess process;

    QString imgName = "img.png";

    process.start("screencapture", QStringList() << "-x" << imgName);
    process.waitForFinished();

    Mat img = imread(imgName.toStdString());

    Rect myROI(70, 350, 500, 500);
    //Mat cropped = img(myROI);
    Mat cropped = img;

    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    //Mat binary ;


    adaptiveThreshold(img, cropped, 250,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 5, 0);

    QPixmap pixmap = QPixmap::fromImage(getImage(cropped));
    QPainter painter(&pixmap);

    painter.setBrush(QBrush(Qt::black));
    QPen pen(QBrush(Qt::black), 3);
    painter.setPen(pen);

    Mat sectors[4][4];
    for(int i = 0; i < 4; ++i)
    {
        QString line;
        for(int j = 0; j < 4; ++j)
        {
            myROI = Rect(10 + 120*j, 10 + 120*i, 110, 110);
            sectors[i][j] = cropped(myROI);
            QString msg = "[" + QString::number(i) + "][" + QString::number(j) + "]: ";
            painter.drawText(10 + 120*j, 10 + 120*i, 50, 20, Qt::AlignLeft, msg);

            QVector<int> hist = getHistogram(sectors[i][j]);
            int cellType = getCellNumber(hist);

            cells[i][j] = cellType;

            QString cellTypeStr = QString::number(cellType);
            painter.drawText(10 + 120*j, 80 + 120*i, 50, 20, Qt::AlignLeft, cellTypeStr);

            line += cellTypeStr + "\t";

            foreach(int val, hist)
            {
                msg += QString::number(val) + " ";
            }
            //logArea->append(msg);

            //QString imgName = "img_" + QString::number(i) + "_" + QString::number(j) + ".jpg";
            //imwrite(imgName.toStdString(), sectors[i][j]);
        }
        logArea->append(line);
    }
    logArea->append("====================");

    gameArea->setPixmap(pixmap);
}

QImage MainWindow::getImage(const Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1)
    {
        // Set the color table (used to translate colour indexes to qRgb values)
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
            colorTable.push_back(qRgb(i,i,i));
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    // 8-bits unsigned, NO. OF CHANNELS=3
    if(mat.type()==CV_8UC3)
    {
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    else
    {
        //logArea->append("Can't convert img");
        //qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}

QVector<int> MainWindow::getHistogram(const Mat mat)
{
    vector<Mat> bgr_planes;
    split(mat, bgr_planes);

    /// Establish the number of bins
    int histSize = 32;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 63;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    QVector<int> hist;
    for( int i = 0; i < histSize; i++ )
    {
        int val = cvRound(r_hist.at<float>(i));
        val += cvRound(g_hist.at<float>(i));
        val += cvRound(b_hist.at<float>(i));

        hist.push_back(val);
//        hist.push_back(cvRound(r_hist.at<float>(i)));
//        hist.push_back(cvRound(g_hist.at<float>(i)));
//        hist.push_back(cvRound(b_hist.at<float>(i)));
    }
    return hist;
}

int MainWindow::getCellNumber(QVector<int> hist)
{
    QMap<int, QVector<int> > cellTypes;
    QVector<int> tmpHist;
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0
            << 0 << 6 << 6 << 63 << 6 << 63 << 63 << 0 << 0 << 0 << 0 << 0 << 0;
    cellTypes.insert(0, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 3 << 3 << 3 << 0 << 0 << 0 << 0
            << 0 << 4 << 4 << 0 << 4 << 0 << 0 << 0 << 63 << 63 << 63 << 0 << 0;
    cellTypes.insert(2, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 3 << 3 << 3 << 0 << 0 << 0 << 0
            << 0 << 4 << 4 << 0 << 4 << 0 << 63 << 0 << 0 << 63 << 63 << 0 << 0;
    cellTypes.insert(4, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 0 << 0
            << 0 << 0 << 4 << 4 << 63 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 70 << 4;
    cellTypes.insert(8, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 0 << 0 << 0 << 0 << 0 << 63
            << 0 << 4 << 4 << 0 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 73 << 5;
    cellTypes.insert(16, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 0 << 0 << 0 << 63 << 0 << 0 << 0
            << 0 << 4 << 4 << 0 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 75 << 6;
    cellTypes.insert(32, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 0 << 0 << 0 << 63 << 0 << 0 << 0 << 0 << 0 << 0 << 0
            << 1 << 3 << 4 << 1 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 77 << 7;
    cellTypes.insert(64, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 0 << 3 << 0 << 4
            << 0 << 0 << 0 << 4 << 0 << 4 << 63 << 3 << 1 << 0 << 63 << 13 << 6;
    cellTypes.insert(128, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 0 << 0 << 2 << 0 << 0 << 5 << 0
            << 0 << 0 << 2 << 3 << 0 << 68 << 3 << 0 << 1 << 64 << 16 << 7;
    cellTypes.insert(256, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 0 << 0 << 0 << 0 << 3 << 4 << 0 << 0 << 0
            << 1 << 0 << 0 << 0 << 4 << 63 << 3 << 5 << 1 << 64 << 14 << 5;
    cellTypes.insert(512, tmpHist);

    tmpHist.clear();
    tmpHist << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 63 << 7 << 2 << 0 << 0 << 0 << 1 << 3 << 0 << 2 << 2 << 0 << 0
            << 0 << 0 << 0 << 2 << 63 << 2 << 5 << 2 << 0 << 63 << 14 << 5;
    cellTypes.insert(1024, tmpHist);

    int histSize = 32;
    int maxDiff = 7;
    foreach(int key, cellTypes.keys())
    {
        bool isEqual = true;
        for(int i = 0; i < histSize; ++i)
        {
            if(abs(cellTypes[key].at(i) - hist.at(i)) > maxDiff)
            {
                isEqual = false;
                break;
            }
        }

        if(isEqual)
            return key;
    }

    return -1;
}

NextCmd MainWindow::getNextCommand()
{
    bool isAvailable = false;
    for(int i = 0; i < 4; ++i)
    {
        if((cells[i][3] != 0 && cells[i][3] == cells[i][2]) ||
                (cells[i][2] != 0 && cells[i][2] == cells[i][1]) ||
                (cells[i][1] != 0 && cells[i][1] == cells[i][0]) ||
                (cells[i][3] != 0 && cells[i][2] == 0 && cells[i][3] == cells[i][1]) ||
                (cells[i][3] != 0 && cells[i][2] == 0 && cells[i][1] == 0 && cells[i][3] == cells[i][0]) ||
                (cells[i][2] != 0 && cells[i][1] == 0 && cells[i][2] == cells[i][0]) )
        {
            isAvailable = true;
            break;
        }
    }

    if(isAvailable)
        return CMD_UP;

//    for(int i = 0; i < 4; ++i)
//    {
//        if((cells[i][3] != 0 && cells[i][3] == cells[i][2]) ||
//                (cells[i][2] != 0 && cells[i][2] == cells[i][1]) ||
//                (cells[i][1] != 0 && cells[i][1] == cells[i][0]) ||
//                (cells[i][3] != 0 && cells[i][2] == 0 && cells[i][3] == cells[i][1]) ||
//                (cells[i][3] != 0 && cells[i][2] == 0 && cells[i][1] == 0 && cells[i][3] == cells[i][0]) ||
//                (cells[i][2] != 0 && cells[i][1] == 0 && cells[i][2] == cells[i][0]) )
//        {
//            isAvailable = true;
//            break;
//        }
//    }

//    if(isAvailable)
//        return CMD_RIGHT;


    for(int i = 0; i < 4; ++i)
    {
        if((cells[i][3] != 0 && cells[i][3] == cells[i][2]) ||
                (cells[i][2] != 0 && cells[i][2] == cells[i][1]) ||
                (cells[i][1] != 0 && cells[i][1] == cells[i][0]) ||
                (cells[i][3] != 0 && cells[i][2] == 0 && cells[i][3] == cells[i][1]) ||
                (cells[i][3] != 0 && cells[i][2] == 0 && cells[i][1] == 0 && cells[i][3] == cells[i][0]) ||
                (cells[i][2] != 0 && cells[i][1] == 0 && cells[i][2] == cells[i][0]) )
        {
            isAvailable = true;
            break;
        }
    }

    if(isAvailable)
        return CMD_DOWN;

    return CMD_UNKNOWN;
}
