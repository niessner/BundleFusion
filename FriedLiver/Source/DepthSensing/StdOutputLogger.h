#ifndef __STDOUTPUTLOGGER__H__
#define __STDOUTPUTLOGGER__H__

#include <windows.h>
#include <Wincon.h>

class StdOutputLogger
{
public:

	enum LogDevice
	{
		LOGDEVICE_CONSOLE,
		LOGDEVICE_FILE,
		LOGDEVICE_NONE
	};

	~StdOutputLogger();

	static void StdOutputLogger::start(LogDevice deviceStdCout, LogDevice deviceStdCerr);
	static void StdOutputLogger::stop();

private:

	StdOutputLogger(LogDevice deviceCout, LogDevice deviceCerr);

	static StdOutputLogger* stdOutputLogger;
};

#endif //__STDOUTPUTLOGGER__H__
