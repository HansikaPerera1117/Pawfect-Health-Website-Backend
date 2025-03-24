import { Module } from '@nestjs/common';
import { ApointmentService } from './apointment.service';
import { ApointmentController } from './apointment.controller';
import { PrismaModule } from 'src/prisma/prisma.module';

@Module({
  imports: [PrismaModule],
  controllers: [ApointmentController],
  providers: [ApointmentService],
})
export class ApointmentModule {}
