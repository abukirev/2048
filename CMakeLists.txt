cmake_minimum_required( VERSION 2.8 )
project( GAME_2048 )
###############################################################################
# Initialization of QT to use it in this project
find_package(Qt4 REQUIRED)
set(QT_USE_QTNETWORK TRUE)
set(QT_USE_QTEST TRUE)

include( ${QT_USE_FILE} )

###############################################################################
set( main_app_src_files
    main.cpp
)

set( src_files
    mainwindow.cpp
)

set( moc_required
    mainwindow.h
)

qt4_wrap_cpp( moc_srcs ${moc_required} )

set( src_files ${src_files} ${moc_srcs} )

include_directories( "/usr/local/include/opencv" )

link_directories( "/usr/local/lib/" )

###############################################################################
add_executable( game_2048 ${main_app_src_files} ${src_files} )
target_link_libraries( game_2048
    ${QT_LIBRARIES}
    "libopencv_core.dylib"
    "libopencv_highgui.dylib"
    "libopencv_imgproc.dylib"
    "libopencv_video.dylib"
    "libopencv_nonfree.dylib"
)
