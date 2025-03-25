import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UserModule } from './user/user.module';
import { PrismaModule } from './prisma/prisma.module';
import { PrismaService } from './prisma/prisma.service';
import { FileModule } from './file/file.module';
import { DoctorModule } from './doctor/doctor.module';
import { ApointmentModule } from './apointment/apointment.module';
import { ChatModule } from './chat/chat.module';

@Module({
  imports: [ UserModule,  PrismaModule, FileModule, DoctorModule, ApointmentModule, ChatModule],
  controllers: [AppController],
  providers: [AppService,PrismaService],
})
export class AppModule {}
