#include <iostream>

#include <QApplication>

#include "mainwindow.h"

using namespace std;

int main(int argc, char* argv[])
{
    QApplication app( argc, argv );
    MainWindow *mw = new MainWindow();
    mw->show();
    app.exec();

    return 0;
}
